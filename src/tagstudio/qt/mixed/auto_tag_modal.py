# SPDX-FileCopyrightText: (c) TagStudio Contributors
# SPDX-License-Identifier: GPL-3.0-only


"""Preview and apply EmbeddingGemma 2 tag suggestions."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, override

import structlog
from PySide6.QtCore import QObject, Qt, QThreadPool, Signal
from PySide6.QtGui import QBrush, QCloseEvent, QColor, QShowEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from tagstudio.core.library.alchemy.tag_verifier import TagVerifier, VerificationConfig
from tagstudio.i18n.translations import Translations
from tagstudio.qt.utils.custom_runnable import CustomRunnable

if TYPE_CHECKING:
    from tagstudio.core.library.alchemy.embeddings import AutoTagResult, SemanticIndex
    from tagstudio.qt.app_settings import AppSettings
    from tagstudio.qt.qt_driver import QtDriver

logger = structlog.get_logger(__name__)

ENTRY_ROLE = int(Qt.ItemDataRole.UserRole)
TAG_ROLE = ENTRY_ROLE + 1
KIND_ROLE = ENTRY_ROLE + 2
NAME_ROLE = ENTRY_ROLE + 3

ENTRY_ROW = "entry"
TAG_ROW = "tag"
EMPTY_ROW = "empty"


class TagSuggestionTask(QObject):
    """Computes tag suggestions for a set of entries in a worker thread."""

    stage = Signal(str)
    progress = Signal(int, int, str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        index: SemanticIndex,
        entry_ids: list[int],
        threshold: float,
        max_per_entry: int,
        one_per_facet: bool,
        use_hubness: bool,
        verifier: TagVerifier | None = None,
    ) -> None:
        super().__init__()
        self.index = index
        self.entry_ids = entry_ids
        self.threshold = threshold
        self.max_per_entry = max_per_entry
        self.one_per_facet = one_per_facet
        self.use_hubness = use_hubness
        self.verifier = verifier
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
            result = self.index.suggest_tags(
                self.entry_ids,
                threshold=self.threshold,
                max_per_entry=self.max_per_entry,
                one_per_facet=self.one_per_facet,
                use_hubness=self.use_hubness,
                verifier=self.verifier,
                stage=self.stage.emit,
                progress=self._on_progress,
                should_cancel=self._cancelled.is_set,
            )
        except Exception as e:
            logger.error("[Embeddings] Could not generate tag suggestions", error=e)
            self.failed.emit(str(e))
            return

        if not self._cancelled.is_set():
            self.finished.emit(result)

    def _on_progress(self, done: int, total: int, label: str) -> None:
        self.progress.emit(done, total, label)


class AutoTagModal(QWidget):
    """Modal that previews automatic tags before applying them.

    Suggestions are grouped per file so the user can see what a single file is
    about to receive, and can drop one tag from every file at once instead of
    unticking it file by file.
    """

    def __init__(self, driver: QtDriver, entry_ids: list[int]) -> None:
        super().__init__()
        self.driver = driver
        self.entry_ids = entry_ids
        self._started = False
        self._task: TagSuggestionTask | None = None
        self._runnable: CustomRunnable | None = None
        self._updating = False

        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setMinimumSize(780, 560)
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

        # Result of the optional vision-language check, hidden until it runs.
        self.verify_label = QLabel()
        self.verify_label.setWordWrap(True)
        self.verify_label.setHidden(True)

        # What the model was asked and what it actually answered. Hidden behind a
        # toggle, but the only way to tell a picky model from a broken one.
        self.verify_details_button = QPushButton(Translations["auto_tag.verify.details"])
        self.verify_details_button.setCheckable(True)
        self.verify_details_button.setHidden(True)
        self.verify_details_button.toggled.connect(self._on_verify_details_toggled)

        self.verify_details_view = QPlainTextEdit()
        self.verify_details_view.setReadOnly(True)
        self.verify_details_view.setHidden(True)
        self.verify_details_view.setMaximumHeight(150)
        self.verify_details_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        details_font = self.verify_details_view.font()
        details_font.setFamilies(["Consolas", "Menlo", "DejaVu Sans Mono", "monospace"])
        self.verify_details_view.setFont(details_font)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(False)

        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText(Translations["auto_tag.filter.placeholder"])
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.textChanged.connect(self._apply_filter)

        self.tree = QTreeWidget()
        self.tree.setRootIsDecorated(True)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tree.setUniformRowHeights(True)
        self.tree.setHeaderLabels(
            [
                Translations["auto_tag.column.apply"],
                Translations["auto_tag.column.item"],
                Translations["auto_tag.column.score"],
            ]
        )
        self.tree.setColumnWidth(0, 64)
        self.tree.setColumnWidth(2, 80)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.tree.itemChanged.connect(self._on_item_changed)
        self.tree.currentItemChanged.connect(lambda *_: self._update_deselect_button())

        self.deselect_tag_button = QPushButton(Translations["auto_tag.deselect_tag"])
        self.deselect_tag_button.setEnabled(False)
        self.deselect_tag_button.setToolTip(Translations["auto_tag.deselect_tag.tooltip"])
        self.deselect_tag_button.clicked.connect(self._deselect_current_tag)

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
        select_layout.addWidget(self.deselect_tag_button)
        select_layout.addStretch(1)
        select_layout.addWidget(self.close_button)
        select_layout.addWidget(self.apply_button)

        self.root_layout.addWidget(self.description_label)
        self.root_layout.addWidget(self.status_label)
        self.root_layout.addWidget(self.verify_label)
        self.root_layout.addWidget(self.verify_details_button, 0, Qt.AlignmentFlag.AlignLeft)
        self.root_layout.addWidget(self.verify_details_view)
        self.root_layout.addWidget(self.progress_bar)
        self.root_layout.addWidget(self.filter_edit)
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
            one_per_facet=settings.auto_tag_one_per_facet,
            use_hubness=settings.auto_tag_hubness,
            verifier=self._build_verifier(settings),
        )
        self._task.progress.connect(self._on_progress)
        self._task.stage.connect(self._on_stage)
        self._task.finished.connect(self._on_finished)
        self._task.failed.connect(self._on_failed)
        self._runnable = CustomRunnable(self._task.run)
        QThreadPool.globalInstance().start(self._runnable)

    def _build_verifier(self, settings: AppSettings) -> TagVerifier | None:
        """Return a verifier when one is switched on and fully configured."""
        if not settings.auto_tag_verify_enabled:
            return None

        config = VerificationConfig(
            base_url=settings.auto_tag_verify_base_url,
            model=settings.auto_tag_verify_model,
            api_key=settings.auto_tag_verify_api_key,
        )
        if not config.is_configured:
            # Switched on but unusable: say so rather than running unchecked and
            # letting it look like the check passed.
            self.verify_label.setText(Translations["auto_tag.verify.unconfigured"])
            self.verify_label.setHidden(False)
            return None

        self.verify_label.setHidden(True)
        return TagVerifier(config)

    def _on_stage(self, stage: str) -> None:
        if stage == "loading_model":
            self.status_label.setText(Translations["auto_tag.loading_model"])
        elif stage == "verifying":
            self.status_label.setText(Translations["auto_tag.verify.stage"])
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

    def _on_finished(self, result: AutoTagResult) -> None:
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)
        self.progress_bar.setHidden(True)

        grouped: dict[int, list] = {}
        names: dict[int, str] = {}
        for suggestion in result.suggestions:
            grouped.setdefault(suggestion.entry_id, []).append(suggestion)
            names[suggestion.entry_id] = suggestion.entry_name
        for entry_id, name in result.unsuggested:
            grouped.setdefault(entry_id, [])
            names[entry_id] = name

        if not grouped:
            self.status_label.setText(Translations["auto_tag.empty"])
            return

        self.status_label.setText(
            Translations.format(
                "auto_tag.found",
                count=len(result.suggestions),
                files=len(grouped),
                empty=len(result.unsuggested),
            )
        )

        self._updating = True
        try:
            for entry_id in sorted(grouped, key=lambda key: names[key].lower()):
                self._add_entry_item(entry_id, names[entry_id], grouped[entry_id])
        finally:
            self._updating = False

        self.tree.expandAll()
        self._update_entry_states()

        has_tags = any(grouped.values())
        self.select_all_button.setEnabled(has_tags)
        self.select_none_button.setEnabled(has_tags)
        self.apply_button.setEnabled(has_tags)

        self._show_verification(result)

    def _show_verification(self, result: AutoTagResult) -> None:
        """Report what the vision check kept, dropped, and could not check."""
        if not result.verify_model:
            return

        text = Translations.format(
            "auto_tag.verify.summary",
            model=result.verify_model,
            kept=len(result.suggestions),
            dropped=result.dropped,
            unchecked=result.unverified,
        )
        if result.verify_error:
            failure = Translations.format("auto_tag.verify.error", error=result.verify_error)
            text = f"{text}\n{failure}"
        self.verify_label.setText(text)
        self.verify_label.setHidden(False)

        self.verify_details_view.setPlainText(self._format_verify_details(result))
        self.verify_details_button.setHidden(not result.verify_notes)
        # A failure is exactly when the raw reply matters, so open the details.
        if result.verify_error:
            self.verify_details_button.setChecked(True)

    def _format_verify_details(self, result: AutoTagResult) -> str:
        """Lay out what each file was asked about and what the model answered."""
        asked = Translations["auto_tag.verify.details.asked"]
        reply = Translations["auto_tag.verify.details.reply"]
        unchecked = Translations["auto_tag.verify.details.unchecked"]
        kept = Translations["auto_tag.verify.details.kept"]

        blocks: list[str] = []
        for note in result.verify_notes:
            lines = [note.entry_name, f"  {asked}: {', '.join(note.labels)}"]
            if note.reply:
                lines.append(f"  {reply}: {note.reply}")
            if note.kept is None:
                lines.append(f"  {Translations.format(unchecked, error=note.error or '-')}")
            else:
                lines.append(
                    f"  {Translations.format(kept, kept=note.kept, sent=len(note.labels))}"
                )
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def _on_verify_details_toggled(self, checked: bool) -> None:
        self.verify_details_view.setVisible(checked)

    def _add_entry_item(self, entry_id: int, name: str, suggestions: list) -> None:
        parent = QTreeWidgetItem(self.tree)
        parent.setData(0, KIND_ROLE, ENTRY_ROW)
        parent.setData(0, ENTRY_ROLE, entry_id)
        parent.setData(0, NAME_ROLE, name)
        parent.setText(1, name)
        parent.setFlags(
            (parent.flags() | Qt.ItemFlag.ItemIsUserCheckable) & ~Qt.ItemFlag.ItemIsEditable
        )

        if not suggestions:
            parent.setCheckState(0, Qt.CheckState.Unchecked)
            parent.setDisabled(True)
            empty = QTreeWidgetItem(parent)
            empty.setText(1, Translations["auto_tag.no_suggestions"])
            empty.setData(0, KIND_ROLE, EMPTY_ROW)
            empty.setDisabled(True)
            return

        parent.setCheckState(0, Qt.CheckState.Checked)
        for suggestion in sorted(suggestions, key=lambda item: -item.score):
            child = QTreeWidgetItem(parent)
            child.setData(0, KIND_ROLE, TAG_ROW)
            child.setData(0, ENTRY_ROLE, entry_id)
            child.setData(0, TAG_ROLE, suggestion.tag_id)
            child.setText(1, suggestion.tag_name)
            child.setText(2, f"{suggestion.score:.3f}")
            child.setTextAlignment(2, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            child.setToolTip(
                1,
                Translations.format(
                    "auto_tag.tag.tooltip", tag=suggestion.tag_name, score=f"{suggestion.score:.3f}"
                ),
            )
            child.setFlags(
                (child.flags() | Qt.ItemFlag.ItemIsUserCheckable) & ~Qt.ItemFlag.ItemIsEditable
            )
            child.setCheckState(0, Qt.CheckState.Checked)

    def _iter_entries(self):
        for index in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(index)
            if item is not None and item.data(0, KIND_ROLE) == ENTRY_ROW:
                yield item

    def _iter_tags(self, parent: QTreeWidgetItem):
        for index in range(parent.childCount()):
            child = parent.child(index)
            if child is not None and child.data(0, KIND_ROLE) == TAG_ROW:
                yield child

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._updating or column != 0:
            return

        state = item.checkState(0)
        self._updating = True
        try:
            if item.data(0, KIND_ROLE) == ENTRY_ROW:
                # Ticking or unticking a file applies to every tag underneath it.
                target = state if state != Qt.CheckState.PartiallyChecked else Qt.CheckState.Checked
                for child in self._iter_tags(item):
                    child.setCheckState(0, target)
            self._update_entry_states()
        finally:
            self._updating = False

    def _update_entry_states(self) -> None:
        """Keep each file's tick and count in step with the tags selected under it."""
        was_updating = self._updating
        self._updating = True
        try:
            for parent in self._iter_entries():
                tags = list(self._iter_tags(parent))
                if not tags:
                    continue
                checked = [tag for tag in tags if tag.checkState(0) == Qt.CheckState.Checked]
                if len(checked) == len(tags):
                    parent.setCheckState(0, Qt.CheckState.Checked)
                elif not checked:
                    parent.setCheckState(0, Qt.CheckState.Unchecked)
                else:
                    parent.setCheckState(0, Qt.CheckState.PartiallyChecked)

                name = str(parent.data(0, NAME_ROLE) or "")
                parent.setText(
                    1,
                    Translations.format(
                        "auto_tag.entry.summary",
                        name=name,
                        selected=len(checked),
                        total=len(tags),
                    ),
                )
                parent.setForeground(
                    1,
                    QBrush(QColor(150, 150, 150)) if not checked else QBrush(),
                )
        finally:
            self._updating = was_updating

    def _update_deselect_button(self) -> None:
        item = self.tree.currentItem()
        if item is not None and item.data(0, KIND_ROLE) == TAG_ROW:
            self.deselect_tag_button.setEnabled(True)
            self.deselect_tag_button.setText(
                Translations.format("auto_tag.deselect_tag.named", tag=item.text(1))
            )
            return

        self.deselect_tag_button.setEnabled(False)
        self.deselect_tag_button.setText(Translations["auto_tag.deselect_tag"])

    def _deselect_current_tag(self) -> None:
        """Untick the selected tag on every file that was offered it."""
        item = self.tree.currentItem()
        if item is None or item.data(0, KIND_ROLE) != TAG_ROW:
            return
        tag_id = item.data(0, TAG_ROLE)

        self._updating = True
        try:
            for parent in self._iter_entries():
                for child in self._iter_tags(parent):
                    if child.data(0, TAG_ROLE) == tag_id:
                        child.setCheckState(0, Qt.CheckState.Unchecked)
            self._update_entry_states()
        finally:
            self._updating = False

        # Keep the row around but visibly dropped, so it is obvious what happened.
        item.setForeground(1, QBrush(QColor(150, 150, 150)))
        self._update_deselect_button()

    def _apply_filter(self, text: str) -> None:
        needle = text.strip().lower()

        for parent in self._iter_entries():
            entry_name = str(parent.data(0, NAME_ROLE) or "").lower()
            parent_matches = needle in entry_name
            any_visible = False
            for child in self._iter_tags(parent):
                matches = not needle or parent_matches or needle in child.text(1).lower()
                child.setHidden(not matches)
                any_visible = any_visible or matches
            parent.setHidden(bool(needle) and not parent_matches and not any_visible)

        if needle:
            self.tree.expandAll()

    def _set_all_checked(self, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        self._updating = True
        try:
            for parent in self._iter_entries():
                if parent.isDisabled():
                    continue
                if not parent.isHidden():
                    parent.setCheckState(0, state)
                for child in self._iter_tags(parent):
                    if not child.isHidden():
                        child.setCheckState(0, state)
            self._update_entry_states()
        finally:
            self._updating = False

    def _on_apply(self) -> None:
        selections: list[tuple[int, int]] = []
        for parent in self._iter_entries():
            for child in self._iter_tags(parent):
                if child.checkState(0) != Qt.CheckState.Checked:
                    continue
                selections.append(
                    (
                        int(child.data(0, ENTRY_ROLE)),
                        int(child.data(0, TAG_ROLE)),
                    )
                )

        if not selections:
            return

        self.driver.apply_auto_tags(selections)
        self.close()

    def _on_failed(self, message: str) -> None:
        logger.error("[Embeddings] Auto-tagging failed", error=message)
        self.progress_bar.setHidden(True)
        self.status_label.setText(f"{Translations['auto_tag.error']}\n{message}")

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
