# SPDX-FileCopyrightText: (c) TagStudio Contributors
# SPDX-License-Identifier: GPL-3.0-only


"""Background task for building the optional semantic search index."""

import threading
from typing import TYPE_CHECKING

import structlog
from PySide6.QtCore import QObject, Signal

from tagstudio.core.library.alchemy.embeddings import IndexReport

if TYPE_CHECKING:
    from tagstudio.core.library.alchemy.embeddings import SemanticIndex

logger = structlog.get_logger(__name__)


class EmbeddingIndexTask(QObject):
    """Builds (or refreshes) the semantic search embeddings in a worker thread."""

    stage = Signal(str)
    progress = Signal(int, int, str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, index: SemanticIndex) -> None:
        super().__init__()
        self.index = index
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        self._cancelled.set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    def run(self) -> None:
        try:
            self.stage.emit("loading_model")
            self.index.prepare()
            if self._cancelled.is_set():
                self.finished.emit(IndexReport())
                return
            self.stage.emit("indexing")
            report = self.index.index_entries(
                progress=self._on_progress,
                should_cancel=self._cancelled.is_set,
            )
        except Exception as e:
            logger.error("[Embeddings] Could not build semantic index", error=e)
            self.failed.emit(str(e))
            return

        self.finished.emit(report)

    def _on_progress(self, done: int, total: int, label: str) -> None:
        self.progress.emit(done, total, label)
