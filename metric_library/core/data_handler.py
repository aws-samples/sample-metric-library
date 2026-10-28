# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from typing import List, Optional
import logging
import os
import json
from pathlib import Path
from .data import DataType, Modality, factory
from .data.text_data import TextData

logger = logging.getLogger(__name__)

# The folders whose data has named a file outside them, each warned about once.
_warned_folders: set[Path] = set()


def extract_nested_values(data, key):
    """
    Extract values for a given key from nested JSON data.

    This function recursively searches through nested dictionaries and lists
    to find all occurrences of a specified key and returns the parent dictionaries
    containing that key.

    Args:
        data: JSON data (dict or list) to search through
        key: Key to extract values for

    Returns:
        List of dictionaries containing the key, or None if not found

    Example:
        >>> data = [{"pred": "A", "gold": "B"}, {"pred": "C", "gold": "C"}]
        >>> extract_nested_values(data, "pred")
        [{"pred": "A", "gold": "B"}, {"pred": "C", "gold": "C"}]
    """
    results = []

    def _extract_recursive(obj):
        if isinstance(obj, dict):
            if key in obj:
                results.append(obj)
            for value in obj.values():
                _extract_recursive(value)
        elif isinstance(obj, list):
            for item in obj:
                _extract_recursive(item)

    _extract_recursive(data)
    return results if results else None


class DataHandler:
    """Handles conversion between raw data and appropriate DataType objects."""

    @staticmethod
    def assign_element(
        element_path: str, modality: Optional[Modality] = None
    ) -> List[DataType]:
        """Create the data item of an explicit path.

        A text path reads its file (a missing path stays the string); an image,
        audio or video path loads through its data type.
        """
        modality = modality if modality else DataHandler._detect_modality(element_path)
        if modality is Modality.TEXT:
            if os.path.isfile(element_path):
                return TextData.from_file(element_path)
            return TextData(element_path)
        return factory.create(modality, element_path)

    @staticmethod
    def create_data(
        path: str,
        modality: Optional[Modality] = None,
        extract_key: Optional[str] = None,
        data_root: Optional[str | os.PathLike] = None,
    ) -> List[DataType]:
        """Create the data items of one path, read without a key unless one is given.

        With ``extract_key`` this is ``parse_json_file``. Without it, an image,
        audio or video path loads as one media item, and any other path that
        is not a file stays the string. A file that holds a non-empty JSON
        array of paths gives one item per entry that names an existing file,
        relative to the folder of ``path`` or absolute; other entries are
        skipped. Any other file, or an array in which no entry names a file,
        is one text item with the content of the file. An error while loading
        a listed file, such as an image that does not open, propagates.

        With ``data_root``, every array entry must lie inside that folder (see
        ``parse_json_file``); ``path`` itself is not checked.
        """
        if extract_key:
            return DataHandler.parse_json_file(path, extract_key, data_root)
        if DataHandler._detect_modality(path) is not Modality.TEXT:
            return [DataHandler.assign_element(path)]
        try:
            with open(path, "r", encoding="utf-8") as inspect_f:
                file_contents = json.load(inspect_f)
        except (
            FileNotFoundError,
            IsADirectoryError,
            NotADirectoryError,
            json.JSONDecodeError,
        ):
            # Not a file: the path stays the string. Not JSON: the content.
            return [DataHandler.assign_element(path, Modality.TEXT)]
        if isinstance(file_contents, list) and file_contents:
            base_dir = Path(path).resolve().parent
            root = DataHandler._resolved_root(data_root)
            list_paths = []
            for index, link in enumerate(file_contents):
                if not isinstance(link, str):
                    continue
                link_path = Path(link)
                if not link_path.is_absolute():
                    link_path = base_dir / link_path
                link_path = DataHandler._inside_root(
                    link_path, link, index, root, base_dir
                )
                if link_path.is_file():
                    list_paths.append(DataHandler.assign_element(str(link_path)))
            if list_paths:
                return list_paths
        return [DataHandler.assign_element(path, Modality.TEXT)]

    @staticmethod
    def _detect_modality(path: str) -> Modality:
        """Detect modality from filepath information for DataType."""

        # Check if file paths
        if isinstance(path, str):
            ext = os.path.splitext(path)[1].lower()

            image_exts = {
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".tiff",
                ".tif",
                ".webp",
                ".gif",
            }
            video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
            audio_exts = {".mp3", ".wav", ".flac", ".ogg", ".aac", ".m4a", ".wma"}

            if ext in image_exts:
                return Modality.IMAGE
            elif ext in video_exts:
                return Modality.VIDEO
            elif ext in audio_exts:
                return Modality.AUDIO
            else:  # making default for text data passed in through a string
                return Modality.TEXT

        # Default fallback
        return None

    @staticmethod
    def batch_create(
        data_items: List[str],
        modality: Optional[Modality] = None,
        extract_keys: Optional[List[str]] = None,
        data_root: Optional[str | os.PathLike] = None,
    ) -> List[List[DataType]]:
        """Create a batch of DataType objects without truncating path/key pairs."""
        if extract_keys is not None:
            if len(data_items) != len(extract_keys):
                raise ValueError(
                    "data_items count "
                    f"({len(data_items)}) must match extract_keys count "
                    f"({len(extract_keys)})"
                )
            return [
                DataHandler.create_data(data_item, modality, extract_key, data_root)
                for data_item, extract_key in zip(data_items, extract_keys)
            ]
        return [
            DataHandler.create_data(data_item, modality, data_root=data_root)
            for data_item in data_items
        ]

    # TODO Remove json parsing here after integration
    @staticmethod
    def parse_json_file(
        path: str,
        extract_key: str,
        data_root: Optional[str | os.PathLike] = None,
    ) -> List[DataType | List[DataType]]:
        """Extract one value per JSON record under ``extract_key``.

        The record's own (top-level) key is used when present; otherwise the key
        must occur exactly once at any depth inside the record.

        A scalar value becomes one data item. A non-empty JSON array of scalars
        becomes a list of items, one per element, so a record can carry several
        items in one role (a candidate answer in several parts, several
        acceptable ground truths). Each element follows the scalar rule: an
        existing image, audio or video path loads as a file, anything else
        stays text. An empty array, or one that holds arrays or objects (an
        empty label set, a list of line items), stays one text item whose raw
        value is the whole array.

        With ``data_root``, every value with an image, audio or video extension
        must lie inside that folder once resolved, symbolic links included,
        whether or not the file exists; else ``ValueError`` names the record.
        ``path`` itself is not checked.
        """
        if extract_key is None or extract_key in {"None", "null"}:
            return []
        if not isinstance(extract_key, str) or not extract_key.strip():
            raise ValueError("extract_key must be a non-empty string")

        with open(path, "r", encoding="utf-8") as file_handle:
            file_data = json.load(file_handle)

        if isinstance(file_data, dict):
            raise ValueError(
                f"Expected a JSON array of records in {path}, got an object "
                f"with keys {list(file_data.keys())}"
            )
        if not isinstance(file_data, list):
            raise ValueError(
                f"Expected a JSON array of records in {path}, got "
                f"{type(file_data).__name__}"
            )

        records = file_data
        if not records:
            raise ValueError(f"No records found while extracting key {extract_key!r}")

        values = []
        for record_index, record in enumerate(records):
            # One lookup rule for every key: a key on the record itself wins,
            # and nested occurrences (a context document with its own ``id``)
            # are ignored. Without a top-level key, the key must occur exactly
            # once anywhere inside the record.
            if isinstance(record, dict) and extract_key in record:
                values.append(record[extract_key])
                continue
            matches = extract_nested_values(record, extract_key) or []
            if len(matches) == 0:
                raise ValueError(
                    f"Missing key {extract_key!r} in record {record_index} of {path}"
                )
            if len(matches) > 1:
                raise ValueError(
                    f"Duplicate key {extract_key!r} in record {record_index} of "
                    f"{path}: found {len(matches)} occurrences"
                )
            values.append(matches[0][extract_key])

        dataset = []
        base_dir = Path(path).resolve().parent
        root = DataHandler._resolved_root(data_root)
        for record_index, value in enumerate(values):
            if (
                isinstance(value, list)
                and value
                and not any(isinstance(part, (list, dict)) for part in value)
            ):
                dataset.append(
                    [
                        DataHandler._scalar_item(part, base_dir, record_index, root)
                        for part in value
                    ]
                )
            else:
                dataset.append(
                    DataHandler._scalar_item(value, base_dir, record_index, root)
                )

        return dataset

    @staticmethod
    def _resolved_root(data_root: Optional[str | os.PathLike]) -> Optional[Path]:
        """The data root with ``~`` expanded and symbolic links resolved, or ``None``."""
        if data_root is None:
            return None
        return Path(os.path.expanduser(data_root)).resolve()

    @staticmethod
    def _inside_root(
        candidate: Path, value: str, index: int, root: Optional[Path], folder: Path
    ) -> Path:
        """Return ``candidate``, resolved when a root confines it; else raise.

        Without a root the candidate is returned unchanged; the first one that
        lies outside ``folder``, the resolved folder of the data file that
        names it, logs a warning for that folder. With a root, its
        resolved path, which follows symbolic links, must lie inside ``root``,
        else ``ValueError`` names entry ``index``. The check comes before any
        test of the file's existence, so a path outside the root fails whether
        or not the file exists. A network (UNC) path that is not on the root's
        own share fails before it is resolved, since resolving it would
        contact its host.
        """
        if root is None:
            if folder not in _warned_folders and not (
                candidate.resolve().is_relative_to(folder)
            ):
                _warned_folders.add(folder)
                logger.warning(
                    "Data in %s names files outside that folder, such as %r; set "
                    "--data-root to confine the files that data files name",
                    folder,
                    value,
                )
            return candidate
        # On Windows a network path's drive is its share, \\host\share; a POSIX
        # path has no drive, so one that starts with // stays an ordinary path.
        share = candidate.drive
        off_share = share.startswith(("\\\\", "//")) and (
            share.lower() != root.drive.lower()
        )
        resolved = None if off_share else candidate.resolve()
        if resolved is None or not resolved.is_relative_to(root):
            raise ValueError(f"record {index}: {value!r} is outside --data-root {root}")
        return resolved

    @staticmethod
    def _scalar_item(
        value, base_dir: Path, record_index: int, root: Optional[Path] = None
    ) -> DataType:
        """One data item for one value: a whole record value or one array element.

        A string with an image, audio or video extension loads as media when it
        names an existing file, relative to the folder of the JSON file or
        absolute. When it does not, it fails the run if the whole string is a
        path (absolute, or relative with a directory separator), and stays text
        otherwise: a bare name or a sentence that ends in ".png" is text. Every
        other value stays text, even a string that names an existing file.
        With a ``root``, the media path must lie inside it (``_inside_root``).
        """
        modality = (
            DataHandler._detect_modality(value) if isinstance(value, str) else None
        )
        if modality is not None and modality is not Modality.TEXT:
            media_path = Path(os.path.expanduser(value))
            if not media_path.is_absolute():
                media_path = base_dir / media_path
            media_path = DataHandler._inside_root(
                media_path, value, record_index, root, base_dir
            )
            if media_path.is_file():
                return DataHandler.assign_element(str(media_path), modality)
            if DataHandler._is_path(value):
                raise ValueError(
                    f"record {record_index}: {modality} file not found: "
                    f"{value!r} (looked for {media_path})"
                )
        return factory.create(Modality.TEXT, value)

    @staticmethod
    def _is_path(value: str) -> bool:
        """Whether the whole string is a file path rather than text.

        A path is absolute (``/...`` or ``~/...``) or relative with a directory
        separator (``renders/a.png``, ``./a.png``), and contains no whitespace.
        A bare file name has no path structure and is not a path.
        """
        if any(character.isspace() for character in value):
            return False
        expanded = os.path.expanduser(value)
        return os.path.isabs(expanded) or "/" in value or os.sep in value
