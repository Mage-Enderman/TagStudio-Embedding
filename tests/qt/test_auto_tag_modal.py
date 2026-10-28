# SPDX-FileCopyrightText: (c) TagStudio Contributors
# SPDX-License-Identifier: GPL-3.0-only

# pyright: reportPrivateUsage=false, reportAttributeAccessIssue=false

from unittest.mock import Mock

from PySide6.QtCore import Qt
from pytestqt.qtbot import QtBot

from tagstudio.core.library.alchemy.embeddings import AutoTagResult, AutoTagSuggestion
from tagstudio.qt.mixed.auto_tag_modal import (
    ENTRY_ROLE,
    ENTRY_ROW,
    KIND_ROLE,
    TAG_ROLE,
    AutoTagModal,
)
from tagstudio.qt.qt_driver import QtDriver


def _suggestion(entry_id: int, name: str, tag_id: int, tag: str, score: float):
    return AutoTagSuggestion(
        entry_id=entry_id,
        entry_name=name,
        tag_id=tag_id,
        tag_name=tag,
        score=score,
    )


def _build_modal(qtbot: QtBot, qt_driver: QtDriver) -> AutoTagModal:
    modal = AutoTagModal(qt_driver, [1, 2, 3])
    qtbot.addWidget(modal)
    return modal


def _result() -> AutoTagResult:
    return AutoTagResult(
        suggestions=[
            _suggestion(1, "PXL_1.mp4", 10, "Beach", 0.82),
            _suggestion(1, "PXL_1.mp4", 11, "Outdoors", 0.79),
            _suggestion(2, "PXL_2.mp4", 10, "Beach", 0.75),
            _suggestion(2, "PXL_2.mp4", 12, "NSFW", 0.74),
        ],
        unsuggested=[(3, "PXL_3.mp4")],
    )


def _entries(modal: AutoTagModal) -> list:
    return [
        modal.tree.topLevelItem(index)
        for index in range(modal.tree.topLevelItemCount())
        if modal.tree.topLevelItem(index).data(0, KIND_ROLE) == ENTRY_ROW
    ]


def test_suggestions_are_grouped_per_file(qtbot: QtBot, qt_driver: QtDriver):
    modal = _build_modal(qtbot, qt_driver)

    modal._on_finished(_result())

    entries = _entries(modal)
    assert len(entries) == 3
    # Files are listed in name order and every requested file appears, including
    # the one that ended up without suggestions.
    assert [entry.data(0, ENTRY_ROLE) for entry in entries] == [1, 2, 3]
    assert entries[0].childCount() == 2
    assert entries[1].childCount() == 2
    assert entries[2].childCount() == 1
    assert entries[2].isDisabled()


def test_entry_row_shows_how_many_tags_are_selected(qtbot: QtBot, qt_driver: QtDriver):
    modal = _build_modal(qtbot, qt_driver)
    modal._on_finished(_result())

    entry = _entries(modal)[0]
    assert entry.checkState(0) == Qt.CheckState.Checked
    assert "2 of 2 selected" in entry.text(1)

    entry.child(0).setCheckState(0, Qt.CheckState.Unchecked)

    assert entry.checkState(0) == Qt.CheckState.PartiallyChecked
    assert "1 of 2 selected" in entry.text(1)


def test_deselecting_a_tag_removes_it_from_every_file(qtbot: QtBot, qt_driver: QtDriver):
    modal = _build_modal(qtbot, qt_driver)
    modal._on_finished(_result())

    entries = _entries(modal)
    nsfw = entries[1].child(1)
    assert nsfw.data(0, TAG_ROLE) == 12

    modal.tree.setCurrentItem(nsfw)
    assert modal.deselect_tag_button.isEnabled()
    modal.deselect_tag_button.click()

    assert nsfw.checkState(0) == Qt.CheckState.Unchecked
    assert "1 of 2 selected" in entries[1].text(1)


def test_unticking_a_file_clears_all_of_its_tags(qtbot: QtBot, qt_driver: QtDriver):
    modal = _build_modal(qtbot, qt_driver)
    modal._on_finished(_result())

    entry = _entries(modal)[0]
    entry.setCheckState(0, Qt.CheckState.Unchecked)

    assert all(
        entry.child(index).checkState(0) == Qt.CheckState.Unchecked
        for index in range(entry.childCount())
    )


def test_apply_only_sends_checked_tags(qtbot: QtBot, qt_driver: QtDriver):
    applied: list[tuple[int, int]] = []
    qt_driver.apply_auto_tags = Mock(side_effect=lambda pairs: applied.extend(pairs))

    modal = _build_modal(qtbot, qt_driver)
    modal._on_finished(_result())

    entries = _entries(modal)
    entries[1].child(1).setCheckState(0, Qt.CheckState.Unchecked)
    modal._on_apply()

    assert sorted(applied) == [(1, 10), (1, 11), (2, 10)]


def test_filter_keeps_matching_files_and_tags_visible(qtbot: QtBot, qt_driver: QtDriver):
    modal = _build_modal(qtbot, qt_driver)
    modal._on_finished(_result())

    modal.filter_edit.setText("nsfw")

    entries = _entries(modal)
    assert entries[0].isHidden()
    assert not entries[1].isHidden()
    assert not entries[1].child(1).isHidden()
    assert entries[1].child(0).isHidden()
