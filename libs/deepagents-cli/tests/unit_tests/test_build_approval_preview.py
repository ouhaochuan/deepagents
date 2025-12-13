import tempfile
import textwrap
from pathlib import Path

import pytest

from deepagents_cli.file_ops import build_approval_preview


def test_build_approval_preview_write_file_new() -> None:
    """Test build_approval_preview for write_file with a new file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target_file = Path(tmpdir) / "new_file.txt"
        
        preview = build_approval_preview(
            "write_file",
            {
                "file_path": str(target_file),
                "content": "Hello World\nThis is a test file.\n",
            },
            assistant_id=None,
        )
        
        assert preview is not None
        assert preview.title == "Write new_file.txt"
        assert preview.details == [
            f"File: {target_file}",
            "Action: Create new file",
            "Lines to write: 2",
        ]
        # Since this is a new file, there's no before content, so diff shows all additions
        assert preview.diff is not None
        assert "+Hello World" in preview.diff
        assert "+This is a test file." in preview.diff
        assert preview.diff_title == "Diff new_file.txt"


def test_build_approval_preview_write_file_overwrite() -> None:
    """Test build_approval_preview for write_file that overwrites an existing file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target_file = Path(tmpdir) / "existing_file.txt"
        target_file.write_text("Old content\n")
        
        preview = build_approval_preview(
            "write_file",
            {
                "file_path": str(target_file),
                "content": "New content\n",
            },
            assistant_id=None,
        )
        
        assert preview is not None
        assert preview.title == "Write existing_file.txt"
        assert "overwrites existing content" in preview.details[1]
        assert preview.diff is not None
        assert "-Old content" in preview.diff
        assert "+New content" in preview.diff


def test_build_approval_preview_edit_file_success() -> None:
    """Test build_approval_preview for edit_file with successful replacement."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target_file = Path(tmpdir) / "test_file.txt"
        target_file.write_text(textwrap.dedent("""\
            First line
            Old content to replace
            Last line
        """))
        
        preview = build_approval_preview(
            "edit_file",
            {
                "file_path": str(target_file),
                "old_string": "Old content to replace",
                "new_string": "New content",
                "replace_all": False,
            },
            assistant_id=None,
        )
        
        assert preview is not None
        assert preview.title == "Update test_file.txt"
        assert preview.details[1] == "Action: Replace text"
        assert "single occurrence" in preview.details[2]
        assert preview.diff is not None
        assert "-Old content to replace" in preview.diff
        assert "+New content" in preview.diff


def test_build_approval_preview_edit_file_replace_all() -> None:
    """Test build_approval_preview for edit_file with replace_all=True."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target_file = Path(tmpdir) / "test_file.txt"
        target_file.write_text(textwrap.dedent("""\
            First line
            Pattern to replace
            Middle line
            Pattern to replace
            Last line
        """))
        
        preview = build_approval_preview(
            "edit_file",
            {
                "file_path": str(target_file),
                "old_string": "Pattern to replace",
                "new_string": "Replaced pattern",
                "replace_all": True,
            },
            assistant_id=None,
        )
        
        assert preview is not None
        assert "all occurrences" in preview.details[2]
        assert preview.diff is not None
        # Should show 2 replacements
        assert preview.diff.count("-Pattern to replace") == 2
        assert preview.diff.count("+Replaced pattern") == 2


def test_build_approval_preview_edit_file_cannot_read() -> None:
    """Test build_approval_preview for edit_file when file cannot be read."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target_file = Path(tmpdir) / "nonexistent.txt"
        
        preview = build_approval_preview(
            "edit_file",
            {
                "file_path": str(target_file),
                "old_string": "something",
                "new_string": "else",
                "replace_all": False,
            },
            assistant_id=None,
        )
        
        assert preview is not None
        assert preview.error == "Unable to read current file contents."


def test_build_approval_preview_edit_file_path_resolution_error() -> None:
    """Test build_approval_preview for edit_file with invalid path."""
    preview = build_approval_preview(
        "edit_file",
        {
            "file_path": None,  # Invalid path
            "old_string": "something",
            "new_string": "else",
            "replace_all": False,
        },
        assistant_id=None,
    )
    
    assert preview is not None
    assert preview.error == "Unable to resolve file path."


def test_build_approval_preview_edit_file_replacement_error() -> None:
    """Test build_approval_preview for edit_file when replacement fails."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target_file = Path(tmpdir) / "test_file.txt"
        target_file.write_text("Some content\n")
        
        preview = build_approval_preview(
            "edit_file",
            {
                "file_path": str(target_file),
                "old_string": "Non-existent content",  # This won't be found
                "new_string": "New content",
                "replace_all": False,
            },
            assistant_id=None,
        )
        
        assert preview is not None
        assert preview.error is not None and "not found in file" in preview.error


def test_build_approval_preview_unknown_tool() -> None:
    """Test build_approval_preview returns None for unknown tool names."""
    preview = build_approval_preview(
        "unknown_tool",
        {"file_path": "some_file.txt"},
        assistant_id=None,
    )
    
    assert preview is None


def test_build_approval_preview_with_assistant_id() -> None:
    """Test build_approval_preview works with assistant_id parameter."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target_file = Path(tmpdir) / "test_file.txt"
        target_file.write_text("Original content\n")
        
        preview = build_approval_preview(
            "write_file",
            {
                "file_path": str(target_file),
                "content": "New content\n",
            },
            assistant_id="test-assistant-id",
        )
        
        assert preview is not None
        assert preview.title == "Write test_file.txt"
        assert preview.diff is not None
        assert "-Original content" in preview.diff
        assert "+New content" in preview.diff