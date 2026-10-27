# SPDX-FileCopyrightText: (c) TagStudio Contributors
# SPDX-License-Identifier: GPL-3.0-only


"""Preview and apply EmbeddingGemma 2 tag suggestions."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, override

import structlog
from PySide6.QtCore import QObject, Qt, QThreadPool, Signal
from PySide6.QtGui import QCloseEvent, QShowEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from tagstudio.i18n.translations import Translations
from tagstudio.qt.utils.custom_runnable import CustomRunnable

if TYPE_CHECKING:
    from tagstudio.core.library.alchemy.embeddings import AutoTagSuggestion, SemanticIndex
    from tagstudio.qt.qt_driver import QtDriver

logger = structlog.get_logger(__name__)


class TagSuggestionTask(QObject):
    """Computes tag suggestions for a set of entries in a worker thread."""

    stage = Signal(str)
    progress = Signal(int, int, str)
    finished = Signal(list)
    failed = Signal(str)

    def __init__(
        self,
        index: SemanticIndex,
        entry_ids: list[int],
        threshold: float,
        max_per_entry: int,
    ) -> None:
        super().__init__()
        self.index = index
        self.entry_ids = entry_ids
        self.threshold = threshold
        self.max_per_entry = max_per_entry
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        self._cancelled.set()

    def run(self) -> None:
        try:
            self.stage.emit("loading_model")
            self.index.prepare()
            if self._cancelled.is_set():
                return
            self.stage.emit("indexing")
            suggestions = self.index.suggest_tags(
                self.entry_ids,
                threshold=self.threshold,
                max_per_entry=self.max_per_entry,
                progress=self._on_progress,
                should_cancel=self._cancelled.is_set,
            )
        except Exception as e:
            logger.error("[Embeddings] Could not generate tag suggestions", error=e)
            self.failed.emit(str(e))
            return

        if not self._cancelled.is_set():
            self.finished.emit(suggestions)

    def _on_progress(self, done: int, total: int, label: str) -> None:
        self.progress.emit(done, total, label)


class AutoTagModal(QWidget):
    """Modal that previews automatic tags before applying them."""

    def __init__(self, driver: QtDriver, entry_ids: list[int]) -> None:
        super().__init__()
        self.driver = driver
        self.entry_ids = entry_ids
        self._started = False
        self._task: TagSuggestionTask | None = None
        self._runnable: CustomRunnable | None = None

        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setMinimumSize(720, 520)
        self.setWindowTitle(
            Translations.format("auto_tag.title.plural", count=len(entry_ids))
            if len(entry_ids) > 1
            else Translations["auto_tag.title.singular"]
        )

        self.root_layout = QVBoxLayout(self)
        self.root_layout.setContentsMargins(12, 12, 12, 12)
        self.root_layout.setSpacing(8)

        self.description_label = QLabel(Translations["auto_tag.description"])
        self.description_label.setWordWrap(True)

        self.status_label = QLabel(Translations["auto_tag.computing"])
        self.status_label.setWordWrap(True)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(False)

        self.tree = QTreeWidget()
        self.tree.setRootIsDecorated(False)
        self.tree.setUniformRowHeights(True)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.tree.setHeaderLabels(
            [
                Translations["auto_tag.column.apply"],
                Translations["auto_tag.column.entry"],
                Translations["auto_tag.column.tag"],
                Translations["auto_tag.column.score"],
            ]
        )
        self.tree.setColumnWidth(0, 64)
        self.tree.setColumnWidth(1, 220)
        self.tree.setColumnWidth(3, 80)
        header = self.tree.header()
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.tree.setColumnHidden(1, len(entry_ids) <= 1)

        self.select_all_button = QPushButton(Translations["auto_tag.select_all"])
        self.select_all_button.clicked.connect(lambda: self._set_all_checked(checked=True))
        self.select_none_button = QPushButton(Translations["auto_tag.select_none"])
        self.select_none_button.clicked.connect(lambda: self._set_all_checked(checked=False))
        self.select_all_button.setEnabled(False)
        self.select_none_button.setEnabled(False)

        self.apply_button = QPushButton(Translations["auto_tag.apply_selected"])
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self._on_apply)

        self.close_button = QPushButton(Translations["generic.close"])
        self.close_button.clicked.connect(self.close)

        select_layout = QHBoxLayout()
        select_layout.addWidget(self.select_all_button)
        select_layout.addWidget(self.select_none_button)
        select_layout.addStretch(1)
        select_layout.addWidget(self.close_button)
        select_layout.addWidget(self.apply_button)

        self.root_layout.addWidget(self.description_label)
        self.root_layout.addWidget(self.status_label)
        self.root_layout.addWidget(self.progress_bar)
        self.root_layout.addWidget(self.tree, 1)
        self.root_layout.addLayout(select_layout)

    def _start(self) -> None:
        index = self.driver.lib.semantic_index
        if index is None:
            self._on_failed(Translations["auto_tag.unavailable.description"])
            return

        settings = self.driver.settings
        self._task = TagSuggestionTask(
            index,
            self.entry_ids,
            threshold=settings.auto_tag_threshold,
            max_per_entry=settings.auto_tag_max_suggestions,
        )
        self._task.progress.connect(self._on_progress)
        self._task.stage.connect(self._on_stage)
        self._task.finished.connect(self._on_finished)
        self._task.failed.connect(self._on_failed)
        self._runnable = CustomRunnable(self._task.run)
        QThreadPool.globalInstance().start(self._runnable)

    def _on_stage(self, stage: str) -> None:
        if stage == "loading_model":
            self.status_label.setText(Translations["auto_tag.loading_model"])
        else:
            self.status_label.setText(Translations["auto_tag.computing"])

    def _on_progress(self, done: int, total: int, label: str) -> None:
        if total > 0:
            if self.progress_bar.maximum() != total:
                self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(done)
        else:
            self.progress_bar.setRange(0, 0)

        if label:
            self.status_label.setText(
                Translations.format(
                    "auto_tag.computing_progress", done=done, total=total, path=label
                )
            )

    def _on_finished(self, suggestions: list[AutoTagSuggestion]) -> None:
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)
        self.progress_bar.setHidden(True)

        if not suggestions:
            self.status_label.setText(Translations["auto_tag.empty"])
            return

        self.status_label.setText(Translations.format("auto_tag.found", count=len(suggestions)))
        for suggestion in suggestions:
            item = QTreeWidgetItem(
                [
                    "",
                    suggestion.entry_name,
                    suggestion.tag_name,
                    f"{suggestion.score:.2f}",
                ]
            )
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(0, Qt.CheckState.Checked)
            item.setData(0, Qt.ItemDataRole.UserRole, suggestion.entry_id)
            item.setData(2, Qt.ItemDataRole.UserRole, suggestion.tag_id)
            self.tree.addTopLevelItem(item)

        self.select_all_button.setEnabled(True)
        self.select_none_button.setEnabled(True)
        self.apply_button.setEnabled(True)

    def _on_failed(self, message: str) -> None:
        logger.error("[Embeddings] Auto-tagging failed", error=message)
        self.progress_bar.setHidden(True)
        self.status_label.setText(f"{Translations['auto_tag.error']}\n{message}")

    def _set_all_checked(self, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            if item is not None:
                item.setCheckState(0, state)

    def _on_apply(self) -> None:
        selections: list[tuple[int, int]] = []
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            if item is None or item.checkState(0) != Qt.CheckState.Checked:
                continue
            selections.append(
                (
                    int(item.data(0, Qt.ItemDataRole.UserRole)),
                    int(item.data(2, Qt.ItemDataRole.UserRole)),
                )
            )

        if not selections:
            return

        self.driver.apply_auto_tags(selections)
        self.close()

    @override
    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        if not self._started:
            self._started = True
            self._start()

    @override
    def closeEvent(self, event: QCloseEvent) -> None:
        if self._task:
            self._task.cancel()
        super().closeEvent(event)
