"""
File utilities for safe atomic writes.
Prevents data corruption on crash mid-write.
"""

import json
import os
import tempfile
import logging

logger = logging.getLogger(__name__)


def atomic_write_json(data_dir: str, filename: str, data: dict) -> bool:
    """
    Atomically write JSON data to disk.
    Writes to a temp file first, then renames to final path.
    If the process crashes mid-write, the original file remains intact.

    Args:
        data_dir: Directory where the file will be stored.
        filename: Name of the target JSON file.
        data: Dictionary to serialize as JSON.

    Returns:
        True on success, False on failure.
    """
    try:
        os.makedirs(data_dir, exist_ok=True)
        target_path = os.path.join(data_dir, filename)
        fd, tmp_path = tempfile.mkstemp(dir=data_dir, suffix='.tmp')
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, target_path)
            return True
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except Exception as e:
        logger.warning(f"Failed to atomic-write {filename}: {e}")
        return False
