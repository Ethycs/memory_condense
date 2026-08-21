"""Race-conscious filesystem primitives for structural corpus packages.

The public corpus verifier never opens a path after merely checking it.  On
POSIX, traversal and file opens are descriptor-relative and use ``O_NOFOLLOW``.
On Windows, every ancestor and entry is opened with
``FILE_FLAG_OPEN_REPARSE_POINT`` and without delete sharing before content is
read.  A tree snapshot keeps those handles alive through verification.
"""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import os
import re
import secrets
import stat
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterator, Mapping

from memory_condense.eval._diffuse_latent_training_corpus_models import (
    MAX_METADATA_FILE_BYTES,
    MAX_PAYLOAD_SHARD_BYTES,
    ROOT_MANIFEST_NAME,
    LatentTrainingCorpusError,
    LatentTrainingFileIdentity,
    _file_identity,
)


_MAX_ROWS = 300
_MAX_FILES = 1 + 2 + _MAX_ROWS + _MAX_ROWS
_MAX_METADATA_FILES = 1 + 2 + _MAX_ROWS
_MAX_AGGREGATE_BYTES = (
    _MAX_ROWS * MAX_PAYLOAD_SHARD_BYTES
    + _MAX_METADATA_FILES * MAX_METADATA_FILE_BYTES
)
_ROW_NAME = re.compile(r"[0-9]{6}\.json\Z")
_PAYLOAD_NAME = re.compile(r"[0-9a-f]{64}\.json\Z")
_DIRECTORIES = ("partitions", "payloads", "rows")


def _error(message: str, exc: BaseException | None = None) -> LatentTrainingCorpusError:
    error = LatentTrainingCorpusError(message)
    if exc is not None:
        error.__cause__ = exc
    return error


def _absolute(value: str | Path) -> Path:
    path = Path(os.path.abspath(os.fspath(value)))
    if not path.name or path.name in {".", ".."}:
        raise LatentTrainingCorpusError("corpus path requires a bounded child name")
    return path


def _file_limit(relative: str) -> int:
    return (
        MAX_PAYLOAD_SHARD_BYTES
        if relative.startswith("payloads/")
        else MAX_METADATA_FILE_BYTES
    )


def _allowed_names(directory: str, names: tuple[str, ...]) -> None:
    if len(names) != len(set(names)):
        raise LatentTrainingCorpusError("corpus enumeration repeated an entry")
    if directory == "":
        if set(names) != {ROOT_MANIFEST_NAME, *_DIRECTORIES}:
            raise LatentTrainingCorpusError("corpus root directory is not closed")
        return
    if directory == "partitions":
        if set(names) != {"fit.json", "validation.json"}:
            raise LatentTrainingCorpusError("partition directory is not closed")
        return
    if directory == "rows":
        if len(names) > _MAX_ROWS or any(_ROW_NAME.fullmatch(name) is None for name in names):
            raise LatentTrainingCorpusError("row directory exceeds its closed schema")
        return
    if directory == "payloads":
        if len(names) > _MAX_ROWS or any(
            _PAYLOAD_NAME.fullmatch(name) is None for name in names
        ):
            raise LatentTrainingCorpusError("payload directory exceeds its closed schema")
        return
    raise LatentTrainingCorpusError("corpus traversal exceeded its maximum depth")


if os.name == "nt":
    from ctypes import wintypes

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _INVALID_HANDLE = ctypes.c_void_p(-1).value
    _GENERIC_READ = 0x80000000
    _GENERIC_WRITE = 0x40000000
    _DELETE = 0x00010000
    _FILE_READ_ATTRIBUTES = 0x0080
    _FILE_LIST_DIRECTORY = 0x0001
    _FILE_SHARE_READ = 0x00000001
    _FILE_SHARE_WRITE = 0x00000002
    _FILE_SHARE_DELETE = 0x00000004
    _OPEN_EXISTING = 3
    _CREATE_NEW = 1
    _FILE_ATTRIBUTE_DIRECTORY = 0x00000010
    _FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400
    _FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
    _FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
    _FILE_NAME_NORMALIZED = 0
    _FILE_RENAME_INFO_CLASS = 3
    _FILE_DISPOSITION_INFO_CLASS = 4

    class _ByHandleFileInformation(ctypes.Structure):
        _fields_ = [
            ("dwFileAttributes", wintypes.DWORD),
            ("ftCreationTime", wintypes.FILETIME),
            ("ftLastAccessTime", wintypes.FILETIME),
            ("ftLastWriteTime", wintypes.FILETIME),
            ("dwVolumeSerialNumber", wintypes.DWORD),
            ("nFileSizeHigh", wintypes.DWORD),
            ("nFileSizeLow", wintypes.DWORD),
            ("nNumberOfLinks", wintypes.DWORD),
            ("nFileIndexHigh", wintypes.DWORD),
            ("nFileIndexLow", wintypes.DWORD),
        ]

    class _FileRenameInformation(ctypes.Structure):
        _fields_ = [
            ("ReplaceIfExists", wintypes.BOOL),
            ("RootDirectory", wintypes.HANDLE),
            ("FileNameLength", wintypes.DWORD),
            ("FileName", wintypes.WCHAR * 1),
        ]

    class _FileDispositionInformation(ctypes.Structure):
        _fields_ = [("DeleteFile", wintypes.BOOL)]

    _kernel32.CreateFileW.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    _kernel32.CreateFileW.restype = wintypes.HANDLE
    _kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    _kernel32.CloseHandle.restype = wintypes.BOOL
    _kernel32.GetFileInformationByHandle.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(_ByHandleFileInformation),
    )
    _kernel32.GetFileInformationByHandle.restype = wintypes.BOOL
    _kernel32.GetFinalPathNameByHandleW.argtypes = (
        wintypes.HANDLE,
        wintypes.LPWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
    )
    _kernel32.GetFinalPathNameByHandleW.restype = wintypes.DWORD
    _kernel32.SetFilePointerEx.argtypes = (
        wintypes.HANDLE,
        ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_longlong),
        wintypes.DWORD,
    )
    _kernel32.SetFilePointerEx.restype = wintypes.BOOL
    _kernel32.ReadFile.argtypes = (
        wintypes.HANDLE,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPVOID,
    )
    _kernel32.ReadFile.restype = wintypes.BOOL
    _kernel32.WriteFile.argtypes = (
        wintypes.HANDLE,
        wintypes.LPCVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPVOID,
    )
    _kernel32.WriteFile.restype = wintypes.BOOL
    _kernel32.FlushFileBuffers.argtypes = (wintypes.HANDLE,)
    _kernel32.FlushFileBuffers.restype = wintypes.BOOL
    _kernel32.SetFileInformationByHandle.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    )
    _kernel32.SetFileInformationByHandle.restype = wintypes.BOOL


if os.name != "nt":
    _libc = ctypes.CDLL(None, use_errno=True)
    _renameat2 = getattr(_libc, "renameat2", None)
    if _renameat2 is not None:
        _renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        _renameat2.restype = ctypes.c_int
    _RENAME_NOREPLACE = 1


@dataclass(frozen=True, slots=True)
class _OpenEntry:
    path: Path
    handle: int
    identity: tuple[int, ...]
    size: int
    is_directory: bool


def _win_error(message: str) -> LatentTrainingCorpusError:
    return _error(message, ctypes.WinError(ctypes.get_last_error()))


def _win_final_path(handle: int) -> str:
    needed = _kernel32.GetFinalPathNameByHandleW(handle, None, 0, _FILE_NAME_NORMALIZED)
    if not needed:
        raise _win_error("cannot resolve an opened corpus handle")
    buffer = ctypes.create_unicode_buffer(needed + 1)
    written = _kernel32.GetFinalPathNameByHandleW(
        handle, buffer, len(buffer), _FILE_NAME_NORMALIZED
    )
    if not written or written >= len(buffer):
        raise _win_error("cannot resolve an opened corpus handle")
    value = buffer.value
    if value.startswith("\\\\?\\UNC\\"):
        value = "\\\\" + value[8:]
    elif value.startswith("\\\\?\\"):
        value = value[4:]
    return os.path.normcase(os.path.normpath(value))


def _win_info(handle: int) -> tuple[tuple[int, ...], int, int]:
    value = _ByHandleFileInformation()
    if not _kernel32.GetFileInformationByHandle(handle, ctypes.byref(value)):
        raise _win_error("cannot inspect an opened corpus handle")
    size = (int(value.nFileSizeHigh) << 32) | int(value.nFileSizeLow)
    write_time = (int(value.ftLastWriteTime.dwHighDateTime) << 32) | int(
        value.ftLastWriteTime.dwLowDateTime
    )
    identity = (
        int(value.dwVolumeSerialNumber),
        int(value.nFileIndexHigh),
        int(value.nFileIndexLow),
        size,
        write_time,
        int(value.dwFileAttributes),
        int(value.nNumberOfLinks),
    )
    return identity, size, int(value.dwFileAttributes)


def _win_open(
    path: Path,
    *,
    directory: bool,
    write: bool = False,
    delete_access: bool = False,
    share_delete: bool = False,
    share_write: bool = False,
) -> _OpenEntry:
    access = _FILE_READ_ATTRIBUTES | (
        _FILE_LIST_DIRECTORY if directory else _GENERIC_READ
    )
    if write:
        access |= _GENERIC_WRITE
    if delete_access:
        access |= _DELETE
    flags = _FILE_FLAG_OPEN_REPARSE_POINT
    if directory:
        flags |= _FILE_FLAG_BACKUP_SEMANTICS
    handle = _kernel32.CreateFileW(
        str(path),
        access,
        _FILE_SHARE_READ
        | (_FILE_SHARE_WRITE if directory or share_write else 0)
        | (_FILE_SHARE_DELETE if share_delete else 0),
        None,
        _OPEN_EXISTING,
        flags,
        None,
    )
    if handle == _INVALID_HANDLE:
        raise _win_error("cannot safely open corpus path")
    raw = int(handle)
    try:
        identity, size, attributes = _win_info(raw)
        if attributes & _FILE_ATTRIBUTE_REPARSE_POINT:
            raise LatentTrainingCorpusError("corpus path contains a reparse point")
        actual_directory = bool(attributes & _FILE_ATTRIBUTE_DIRECTORY)
        if actual_directory is not directory:
            raise LatentTrainingCorpusError("corpus entry has the wrong filesystem type")
        if _win_final_path(raw) != os.path.normcase(os.path.normpath(str(path))):
            raise LatentTrainingCorpusError("opened corpus path escaped its lexical location")
        return _OpenEntry(path, raw, identity, size, directory)
    except BaseException:
        _kernel32.CloseHandle(raw)
        raise


def _win_close(handle: int) -> None:
    if not _kernel32.CloseHandle(handle):
        raise _win_error("cannot close corpus handle")


def _win_rename(entry: _OpenEntry, parent: _OpenEntry, name: str) -> None:
    destination = str(parent.path / name)
    encoded = destination.encode("utf-16-le")
    # Allocate the flexible array with a trailing WCHAR even though the API's
    # FileNameLength deliberately excludes that terminator.
    size = ctypes.sizeof(_FileRenameInformation) + len(encoded)
    buffer = ctypes.create_string_buffer(size)
    value = ctypes.cast(buffer, ctypes.POINTER(_FileRenameInformation)).contents
    value.ReplaceIfExists = False
    value.RootDirectory = None
    value.FileNameLength = len(encoded)
    ctypes.memmove(
        ctypes.addressof(buffer) + _FileRenameInformation.FileName.offset,
        encoded,
        len(encoded),
    )
    if not _kernel32.SetFileInformationByHandle(
        entry.handle,
        _FILE_RENAME_INFO_CLASS,
        buffer,
        size,
    ):
        code = ctypes.get_last_error()
        if code in {80, 183}:
            raise FileExistsError(parent.path / name)
        raise _win_error("cannot atomically publish corpus directory")


def _win_mark_delete(entry: _OpenEntry) -> None:
    value = _FileDispositionInformation(True)
    if not _kernel32.SetFileInformationByHandle(
        entry.handle,
        _FILE_DISPOSITION_INFO_CLASS,
        ctypes.byref(value),
        ctypes.sizeof(value),
    ):
        raise _win_error("cannot delete owned corpus object by handle")


def _win_read(entry: _OpenEntry, limit: int) -> bytes:
    if entry.is_directory or entry.size > limit:
        raise LatentTrainingCorpusError("corpus file exceeds its pre-parse byte cap")
    if not _kernel32.SetFilePointerEx(entry.handle, 0, None, 0):
        raise _win_error("cannot seek corpus file")
    chunks: list[bytes] = []
    remaining = entry.size
    while remaining:
        width = min(remaining, 1024 * 1024)
        buffer = ctypes.create_string_buffer(width)
        read = wintypes.DWORD()
        if not _kernel32.ReadFile(
            entry.handle, buffer, width, ctypes.byref(read), None
        ):
            raise _win_error("cannot read corpus file")
        if read.value == 0:
            raise LatentTrainingCorpusError("corpus file ended before its snapshotted size")
        chunks.append(buffer.raw[: read.value])
        remaining -= read.value
    payload = b"".join(chunks)
    if _win_info(entry.handle)[0] != entry.identity:
        raise LatentTrainingCorpusError("corpus file changed while being read")
    return payload


def _posix_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _same_object(left: tuple[int, ...], right: tuple[int, ...]) -> bool:
    if os.name == "nt":
        return left[:3] == right[:3]
    return left[:2] == right[:2] and stat.S_IFMT(left[2]) == stat.S_IFMT(right[2])


def _posix_flags(*, directory: bool, write: bool = False, create: bool = False) -> int:
    flags = os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    if directory:
        return flags | os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    if create:
        return flags | os.O_RDWR | os.O_CREAT | os.O_EXCL
    # A hostile pathname may name a FIFO or device.  Nonblocking open lets us
    # reach fstat/type rejection without waiting on an external peer.
    return flags | getattr(os, "O_NONBLOCK", 0) | (
        os.O_RDWR if write else os.O_RDONLY
    )


def _posix_open_at(parent_fd: int, name: str, path: Path, *, directory: bool) -> _OpenEntry:
    try:
        fd = os.open(name, _posix_flags(directory=directory), dir_fd=parent_fd)
        value = os.fstat(fd)
    except OSError as exc:
        raise _error("cannot safely open corpus path", exc) from exc
    expected = stat.S_ISDIR(value.st_mode) if directory else stat.S_ISREG(value.st_mode)
    if not expected:
        os.close(fd)
        raise LatentTrainingCorpusError("corpus entry has the wrong filesystem type")
    return _OpenEntry(path, fd, _posix_identity(value), int(value.st_size), directory)


def _posix_rename_noreplace(parent: _OpenEntry, source: str, target: str) -> None:
    if _renameat2 is None:
        raise LatentTrainingCorpusError(
            "atomic no-replace directory publication is unavailable"
        )
    result = _renameat2(
        parent.handle,
        os.fsencode(source),
        parent.handle,
        os.fsencode(target),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        code = ctypes.get_errno()
        if code == getattr(os, "EEXIST", 17):
            raise FileExistsError(parent.path / target)
        raise _error(
            "cannot atomically publish corpus directory",
            OSError(code, os.strerror(code)),
        )


def _open_chain(path: Path, *, leaf_delete: bool = False) -> list[_OpenEntry]:
    absolute = _absolute(path)
    if os.name == "nt":
        chain: list[_OpenEntry] = []
        anchor = Path(absolute.anchor)
        pieces = absolute.parts[1:]
        current = anchor
        try:
            chain.append(_win_open(current, directory=True))
            for index, piece in enumerate(pieces):
                current = current / piece
                chain.append(
                    _win_open(
                        current,
                        directory=True,
                        delete_access=leaf_delete and index == len(pieces) - 1,
                    )
                )
            return chain
        except BaseException:
            for entry in reversed(chain):
                _win_close(entry.handle)
            raise
    chain = []
    try:
        fd = os.open(absolute.anchor, _posix_flags(directory=True))
        root_info = os.fstat(fd)
        chain.append(
            _OpenEntry(Path(absolute.anchor), fd, _posix_identity(root_info), 0, True)
        )
        current = Path(absolute.anchor)
        for piece in absolute.parts[1:]:
            current = current / piece
            chain.append(_posix_open_at(chain[-1].handle, piece, current, directory=True))
        return chain
    except BaseException:
        for entry in reversed(chain):
            os.close(entry.handle)
        raise


def _close_entry(entry: _OpenEntry) -> None:
    if os.name == "nt":
        _win_close(entry.handle)
    else:
        os.close(entry.handle)


def _open_child(
    parent: _OpenEntry,
    name: str,
    *,
    directory: bool,
    delete_access: bool = False,
) -> _OpenEntry:
    path = parent.path / name
    if os.name == "nt":
        return _win_open(path, directory=directory, delete_access=delete_access)
    return _posix_open_at(parent.handle, name, path, directory=directory)


def _directory_entry_cap(directory: str) -> int:
    return {"": 4, "partitions": 2, "rows": _MAX_ROWS, "payloads": _MAX_ROWS}[
        directory
    ]


def _entry_names(directory: _OpenEntry, *, cap: int) -> tuple[str, ...]:
    try:
        values = []
        with os.scandir(directory.path if os.name == "nt" else directory.handle) as stream:
            for entry in stream:
                if len(values) >= cap:
                    raise LatentTrainingCorpusError(
                        "corpus directory exceeds its entry-count cap"
                    )
                values.append(entry.name)
    except OSError as exc:
        raise _error("cannot enumerate corpus package", exc) from exc
    if any(type(item) is not str or not item or item in {".", ".."} for item in values):
        raise LatentTrainingCorpusError("corpus enumeration returned an invalid name")
    return tuple(values)


def _read_entry(entry: _OpenEntry, limit: int) -> bytes:
    if os.name == "nt":
        return _win_read(entry, limit)
    if entry.size > limit:
        raise LatentTrainingCorpusError("corpus file exceeds its pre-parse byte cap")
    try:
        os.lseek(entry.handle, 0, os.SEEK_SET)
        remaining = entry.size
        chunks = []
        while remaining:
            value = os.read(entry.handle, min(remaining, 1024 * 1024))
            if not value:
                raise LatentTrainingCorpusError(
                    "corpus file ended before its snapshotted size"
                )
            chunks.append(value)
            remaining -= len(value)
        payload = b"".join(chunks)
        if _posix_identity(os.fstat(entry.handle)) != entry.identity:
            raise LatentTrainingCorpusError("corpus file changed while being read")
        return payload
    except OSError as exc:
        raise _error("cannot read corpus file", exc) from exc


@dataclass(frozen=True, slots=True)
class CorpusFileSnapshot:
    relative_path: str
    size: int
    sha256: str
    _entry: _OpenEntry


@dataclass(frozen=True, slots=True)
class OwnedCorpusDirectory:
    """Unforgeable-in-practice identity carried across publication phases."""

    path: Path
    parent: Path
    prefix: str
    identity: tuple[int, ...]
    child_identities: tuple[tuple[str, tuple[int, ...]], ...]

    def __fspath__(self) -> str:
        return os.fspath(self.path)


class CorpusTreeSnapshot:
    """Held-handle, immutable view of one closed corpus tree."""

    __slots__ = (
        "root",
        "files",
        "directories",
        "_ancestors",
        "_directory_entries",
        "_initial_names",
        "_closed",
    )

    def __init__(self, root: str | Path) -> None:
        self.root = _absolute(root)
        self._ancestors = _open_chain(self.root)
        self._directory_entries: dict[str, _OpenEntry] = {"": self._ancestors[-1]}
        self._initial_names: dict[str, frozenset[str]] = {}
        self._closed = False
        opened_files: dict[str, CorpusFileSnapshot] = {}
        total_bytes = 0
        try:
            root_names = _entry_names(
                self._directory_entries[""], cap=_directory_entry_cap("")
            )
            _allowed_names("", root_names)
            self._initial_names[""] = frozenset(root_names)
            for name in _DIRECTORIES:
                child = _open_child(self._directory_entries[""], name, directory=True)
                self._directory_entries[name] = child
                names = _entry_names(child, cap=_directory_entry_cap(name))
                _allowed_names(name, names)
                self._initial_names[name] = frozenset(names)
            file_names = [("", ROOT_MANIFEST_NAME)]
            file_names.extend(("partitions", name) for name in self._initial_names["partitions"])
            file_names.extend(("rows", name) for name in self._initial_names["rows"])
            file_names.extend(("payloads", name) for name in self._initial_names["payloads"])
            if len(file_names) > _MAX_FILES:
                raise LatentTrainingCorpusError("corpus tree exceeds its file-count cap")
            for directory, name in sorted(file_names):
                relative = name if not directory else f"{directory}/{name}"
                entry = _open_child(self._directory_entries[directory], name, directory=False)
                limit = _file_limit(relative)
                if entry.size > limit:
                    _close_entry(entry)
                    raise LatentTrainingCorpusError("corpus file exceeds its pre-parse byte cap")
                total_bytes += entry.size
                if total_bytes > _MAX_AGGREGATE_BYTES:
                    _close_entry(entry)
                    raise LatentTrainingCorpusError("corpus tree exceeds its aggregate byte cap")
                payload = _read_entry(entry, limit)
                opened_files[relative] = CorpusFileSnapshot(
                    relative, entry.size, hashlib.sha256(payload).hexdigest(), entry
                )
            self.files = MappingProxyType(dict(opened_files))
            self.directories = frozenset(_DIRECTORIES)
        except BaseException:
            for snapshot in opened_files.values():
                _close_entry(snapshot._entry)
            for name, entry in reversed(tuple(self._directory_entries.items())):
                if name:
                    _close_entry(entry)
            for entry in reversed(self._ancestors):
                _close_entry(entry)
            self._closed = True
            raise

    def read(self, relative: str) -> bytes:
        if self._closed:
            raise RuntimeError("corpus snapshot is closed")
        if type(relative) is not str or relative not in self.files:
            raise LatentTrainingCorpusError("requested file is outside the corpus snapshot")
        snapshot = self.files[relative]
        payload = _read_entry(snapshot._entry, _file_limit(relative))
        if len(payload) != snapshot.size or hashlib.sha256(payload).hexdigest() != snapshot.sha256:
            raise LatentTrainingCorpusError("corpus file changed during verification")
        return payload

    def assert_unchanged(self) -> None:
        if self._closed:
            raise RuntimeError("corpus snapshot is closed")
        for name, directory in self._directory_entries.items():
            if frozenset(
                _entry_names(directory, cap=_directory_entry_cap(name))
            ) != self._initial_names[name]:
                raise LatentTrainingCorpusError("corpus entries changed during verification")
            if os.name == "nt":
                current = _win_info(directory.handle)[0]
                if _win_final_path(directory.handle) != os.path.normcase(
                    os.path.normpath(str(directory.path))
                ):
                    raise LatentTrainingCorpusError("corpus directory moved during verification")
            else:
                current = _posix_identity(os.fstat(directory.handle))
            if current != directory.identity:
                raise LatentTrainingCorpusError("corpus directory changed during verification")
        root_entry = self._directory_entries[""]
        _assert_named_object(self._ancestors[-2], self.root.name, root_entry)
        for name in _DIRECTORIES:
            _assert_named_object(root_entry, name, self._directory_entries[name])
        for relative, snapshot in self.files.items():
            parent_name, name = relative.rsplit("/", 1) if "/" in relative else ("", relative)
            current = _open_child(self._directory_entries[parent_name], name, directory=False)
            try:
                if current.identity != snapshot._entry.identity:
                    raise LatentTrainingCorpusError("corpus file was replaced during verification")
            finally:
                _close_entry(current)
            self.read(relative)

    def close(self) -> None:
        if self._closed:
            return
        errors = []
        for snapshot in reversed(tuple(self.files.values())):
            try:
                _close_entry(snapshot._entry)
            except BaseException as exc:
                errors.append(exc)
        for name, entry in reversed(tuple(self._directory_entries.items())):
            if name:
                try:
                    _close_entry(entry)
                except BaseException as exc:
                    errors.append(exc)
        for entry in reversed(self._ancestors):
            try:
                _close_entry(entry)
            except BaseException as exc:
                errors.append(exc)
        self._closed = True
        if errors:
            raise LatentTrainingCorpusError("cannot release corpus snapshot handles") from errors[0]

    def __enter__(self) -> "CorpusTreeSnapshot":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def require_plain_parent(path: str | Path) -> Path:
    absolute = _absolute(path)
    chain = _open_chain(absolute)
    try:
        return absolute
    finally:
        for entry in reversed(chain):
            _close_entry(entry)


def owned_staging(parent: Path, name: str) -> OwnedCorpusDirectory:
    parent = _absolute(parent)
    chain = _open_chain(parent)
    parent_entry = chain[-1]
    staging = None
    try:
        for _ in range(128):
            candidate_name = f".{name}.staging-{secrets.token_hex(8)}"
            candidate = parent / candidate_name
            try:
                if os.name == "nt":
                    candidate.mkdir(mode=0o700)
                else:
                    os.mkdir(candidate_name, mode=0o700, dir_fd=parent_entry.handle)
            except FileExistsError:
                continue
            staging = candidate
            break
        if staging is None:
            raise FileExistsError("cannot allocate unique corpus staging directory")
        staging_entry = _open_child(parent_entry, staging.name, directory=True)
        child_entries: list[_OpenEntry] = []
        try:
            for child in _DIRECTORIES:
                if os.name == "nt":
                    (staging / child).mkdir()
                else:
                    os.mkdir(child, mode=0o700, dir_fd=staging_entry.handle)
                child_entries.append(
                    _open_child(staging_entry, child, directory=True)
                )
            _flush_open_directory(staging_entry)
            _flush_open_directory(parent_entry)
            owned = OwnedCorpusDirectory(
                staging,
                parent,
                f".{name}.staging-",
                staging_entry.identity,
                tuple((entry.path.name, entry.identity) for entry in child_entries),
            )
        finally:
            for entry in reversed(child_entries):
                _close_entry(entry)
            _close_entry(staging_entry)
        return owned
    except BaseException:
        if staging is not None:
            try:
                entry = _open_child(parent_entry, staging.name, directory=True)
            except BaseException:
                pass
            else:
                children = []
                try:
                    for child_name in _DIRECTORIES:
                        try:
                            children.append(
                                _open_child(entry, child_name, directory=True)
                            )
                        except BaseException:
                            pass
                    child_identities = tuple(
                        (child.path.name, child.identity) for child in children
                    )
                finally:
                    for child in reversed(children):
                        _close_entry(child)
                token = OwnedCorpusDirectory(
                    staging,
                    parent,
                    f".{name}.staging-",
                    entry.identity,
                    child_identities,
                )
                _close_entry(entry)
                remove_owned(token)
        raise
    finally:
        for entry in reversed(chain):
            _close_entry(entry)


@contextlib.contextmanager
def _safe_parent(path: Path) -> Iterator[_OpenEntry]:
    chain = _open_chain(path)
    try:
        yield chain[-1]
    finally:
        for entry in reversed(chain):
            _close_entry(entry)


def _existing_bytes_at(parent: _OpenEntry, name: str, limit: int) -> bytes:
    entry = _open_child(parent, name, directory=False)
    try:
        if entry.size > limit:
            raise LatentTrainingCorpusError("content-addressed collision exceeds byte cap")
        return _read_entry(entry, limit)
    finally:
        _close_entry(entry)


def _write_windows(path: Path, payload: bytes) -> None:
    handle = _kernel32.CreateFileW(
        str(path),
        _GENERIC_WRITE | _GENERIC_READ | _FILE_READ_ATTRIBUTES,
        _FILE_SHARE_READ,
        None,
        _CREATE_NEW,
        _FILE_FLAG_OPEN_REPARSE_POINT,
        None,
    )
    if handle == _INVALID_HANDLE:
        code = ctypes.get_last_error()
        if code in {80, 183}:
            raise FileExistsError(path)
        raise _win_error("cannot create corpus file")
    raw = int(handle)
    try:
        offset = 0
        while offset < len(payload):
            chunk = payload[offset : offset + 1024 * 1024]
            written = wintypes.DWORD()
            if not _kernel32.WriteFile(raw, chunk, len(chunk), ctypes.byref(written), None):
                raise _win_error("cannot write corpus file")
            if written.value != len(chunk):
                raise LatentTrainingCorpusError("short corpus file write")
            offset += written.value
        if not _kernel32.FlushFileBuffers(raw):
            raise _win_error("cannot flush corpus file")
    finally:
        _win_close(raw)


def _assert_named_object(
    parent: _OpenEntry,
    name: str,
    expected: _OpenEntry,
) -> None:
    current = _open_child(parent, name, directory=expected.is_directory)
    try:
        if not _same_object(current.identity, expected.identity):
            raise LatentTrainingCorpusError("owned corpus pathname was replaced")
    finally:
        _close_entry(current)


@contextlib.contextmanager
def _open_owned(
    owned: OwnedCorpusDirectory,
) -> Iterator[tuple[list[_OpenEntry], _OpenEntry, Mapping[str, _OpenEntry]]]:
    if type(owned) is not OwnedCorpusDirectory or not _tree_is_owned(
        owned.path, owned.parent, owned.prefix
    ):
        raise RuntimeError("refusing an invalid corpus ownership capability")
    chain = _open_chain(owned.path)
    children: dict[str, _OpenEntry] = {}
    try:
        root = chain[-1]
        if not _same_object(root.identity, owned.identity):
            raise LatentTrainingCorpusError("owned corpus pathname was replaced")
        _assert_named_object(chain[-2], owned.path.name, root)
        expected_children = dict(owned.child_identities)
        if len(expected_children) != len(owned.child_identities) or any(
            name not in _DIRECTORIES for name in expected_children
        ):
            raise RuntimeError("owned corpus child capabilities are malformed")
        for name, identity in expected_children.items():
            child = _open_child(root, name, directory=True)
            if not _same_object(child.identity, identity):
                _close_entry(child)
                raise LatentTrainingCorpusError("owned corpus child was replaced")
            children[name] = child
        yield chain, root, MappingProxyType(dict(children))
    finally:
        for entry in reversed(tuple(children.values())):
            _close_entry(entry)
        for entry in reversed(chain):
            _close_entry(entry)


def write_new(
    root: OwnedCorpusDirectory,
    relative: str,
    payload: bytes,
) -> LatentTrainingFileIdentity:
    if type(payload) is not bytes or type(relative) is not str:
        raise TypeError("corpus writes require exact bytes and relative strings")
    parts = relative.split("/")
    if len(parts) == 1:
        if parts[0] != ROOT_MANIFEST_NAME:
            raise LatentTrainingCorpusError("invalid root corpus write")
    elif len(parts) == 2:
        directory, name = parts
        _allowed_names(directory, (name,)) if directory not in {"partitions"} else None
        if directory == "partitions" and name not in {"fit.json", "validation.json"}:
            raise LatentTrainingCorpusError("invalid partition corpus write")
    else:
        raise LatentTrainingCorpusError("corpus writes cannot create nested paths")
    limit = _file_limit(relative)
    if len(payload) > limit:
        raise LatentTrainingCorpusError("corpus write exceeds its byte cap")
    with _open_owned(root) as (_, root_entry, children):
        parent = root_entry
        if len(parts) == 2:
            if parts[0] not in children:
                raise LatentTrainingCorpusError("corpus child capability is missing")
            parent = children[parts[0]]
        target = parent.path / parts[-1]
        if len(parts) == 2:
            _assert_named_object(root_entry, parts[0], parent)
        if os.name == "nt":
            try:
                _write_windows(target, payload)
            except FileExistsError:
                if not relative.startswith("payloads/"):
                    raise LatentTrainingCorpusError(
                        "non-payload corpus files are strictly no-clobber"
                    )
                if _existing_bytes_at(parent, target.name, limit) != payload:
                    raise LatentTrainingCorpusError("content-addressed file collision")
        else:
            try:
                fd = os.open(
                    target.name,
                    _posix_flags(directory=False, create=True),
                    0o600,
                    dir_fd=parent.handle,
                )
            except FileExistsError:
                if not relative.startswith("payloads/"):
                    raise LatentTrainingCorpusError(
                        "non-payload corpus files are strictly no-clobber"
                    )
                if _existing_bytes_at(parent, target.name, limit) != payload:
                    raise LatentTrainingCorpusError("content-addressed file collision")
            else:
                try:
                    view = memoryview(payload)
                    while view:
                        written = os.write(fd, view)
                        if written <= 0:
                            raise LatentTrainingCorpusError("short corpus file write")
                        view = view[written:]
                    os.fsync(fd)
                finally:
                    os.close(fd)
        _flush_open_directory(parent)
    return _file_identity(relative, payload)


def _flush_open_directory(entry: _OpenEntry) -> None:
    if os.name == "nt":
        writable = _win_open(entry.path, directory=True, write=True)
        try:
            if not _same_object(writable.identity, entry.identity):
                raise LatentTrainingCorpusError("directory changed before flush")
            if not _kernel32.FlushFileBuffers(writable.handle):
                raise _win_error("cannot flush corpus directory")
        finally:
            _win_close(writable.handle)
    else:
        try:
            os.fsync(entry.handle)
        except OSError as exc:
            raise _error("cannot flush corpus directory", exc) from exc


def _tree_is_owned(path: Path, parent: Path, prefix: str) -> bool:
    return path.parent == parent and path.name.startswith(prefix)


def remove_owned(owned: OwnedCorpusDirectory) -> None:
    if type(owned) is not OwnedCorpusDirectory or not _tree_is_owned(
        owned.path, owned.parent, owned.prefix
    ):
        raise RuntimeError("refusing to remove an unowned corpus path")
    path = owned.path
    if not os.path.lexists(path):
        return
    # Preflight the complete tree and keep every object handle live before the
    # first mutation.  If anything is malformed or replaced, nothing is
    # removed.  POSIX deletion remains descriptor-relative with the checked
    # object open; Windows handles pin names until each exact deletion.
    try:
        chain = _open_chain(path, leaf_delete=os.name == "nt")
    except BaseException:
        return
    root = chain[-1]
    parent_entry = chain[-2]
    directories: dict[str, _OpenEntry] = {}
    files: dict[tuple[str, str], _OpenEntry] = {}
    closed: set[int] = set()

    def close_once(entry: _OpenEntry) -> None:
        if entry.handle not in closed:
            _close_entry(entry)
            closed.add(entry.handle)

    def assert_current(parent: _OpenEntry, name: str, entry: _OpenEntry) -> None:
        if os.name == "nt":
            if _win_final_path(entry.handle) != os.path.normcase(
                os.path.normpath(str(entry.path))
            ):
                raise LatentTrainingCorpusError("owned corpus pathname was replaced")
        else:
            _assert_named_object(parent, name, entry)

    try:
        if not _same_object(root.identity, owned.identity):
            raise LatentTrainingCorpusError("owned corpus pathname was replaced")
        assert_current(parent_entry, path.name, root)
        root_names = set(_entry_names(root, cap=4))
        if not root_names <= {ROOT_MANIFEST_NAME, *_DIRECTORIES}:
            raise LatentTrainingCorpusError(
                "refusing to remove a changed owned corpus tree"
            )
        if (root_names & set(_DIRECTORIES)) != {
            name for name, _ in owned.child_identities
        }:
            raise LatentTrainingCorpusError(
                "owned corpus child capabilities no longer match the tree"
            )
        for directory_name in _DIRECTORIES:
            if directory_name not in root_names:
                continue
            child = _open_child(
                root,
                directory_name,
                directory=True,
                delete_access=os.name == "nt",
            )
            expected_child = dict(owned.child_identities).get(directory_name)
            if expected_child is None or not _same_object(
                child.identity, expected_child
            ):
                _close_entry(child)
                raise LatentTrainingCorpusError("owned corpus child was replaced")
            directories[directory_name] = child
            names = _entry_names(
                child,
                cap=2 if directory_name == "partitions" else _MAX_ROWS,
            )
            if directory_name == "partitions" and not set(names) <= {
                "fit.json", "validation.json"
            }:
                raise LatentTrainingCorpusError(
                    "refusing to remove a changed partition directory"
                )
            if directory_name == "rows" and any(
                _ROW_NAME.fullmatch(name) is None for name in names
            ):
                raise LatentTrainingCorpusError(
                    "refusing to remove a changed row directory"
                )
            if directory_name == "payloads" and any(
                _PAYLOAD_NAME.fullmatch(name) is None for name in names
            ):
                raise LatentTrainingCorpusError(
                    "refusing to remove a changed payload directory"
                )
            for name in names:
                files[(directory_name, name)] = _open_child(
                    child,
                    name,
                    directory=False,
                    delete_access=os.name == "nt",
                )
        if ROOT_MANIFEST_NAME in root_names:
            files[("", ROOT_MANIFEST_NAME)] = _open_child(
                root,
                ROOT_MANIFEST_NAME,
                directory=False,
                delete_access=os.name == "nt",
            )

        # Rebind all public names after the complete preflight and before the
        # first deletion.  This also detects entries added during preflight.
        assert_current(parent_entry, path.name, root)
        if set(_entry_names(root, cap=4)) != root_names:
            raise LatentTrainingCorpusError("owned corpus changed during cleanup preflight")
        for name, child in directories.items():
            assert_current(root, name, child)
            expected = {file_name for directory, file_name in files if directory == name}
            if set(_entry_names(child, cap=2 if name == "partitions" else _MAX_ROWS)) != expected:
                raise LatentTrainingCorpusError("owned corpus changed during cleanup preflight")
        for (directory, name), entry in files.items():
            assert_current(root if not directory else directories[directory], name, entry)

        for (directory, name), entry in files.items():
            parent = root if not directory else directories[directory]
            assert_current(parent, name, entry)
            if os.name == "nt":
                _win_mark_delete(entry)
                close_once(entry)
            else:
                os.unlink(name, dir_fd=parent.handle)
                close_once(entry)
        for directory_name, child in directories.items():
            assert_current(root, directory_name, child)
            if os.name == "nt":
                _win_mark_delete(child)
                close_once(child)
            else:
                os.rmdir(directory_name, dir_fd=root.handle)
                close_once(child)
        if os.name != "nt":
            _flush_open_directory(root)
        assert_current(parent_entry, path.name, root)
        if os.name == "nt":
            _win_mark_delete(root)
            close_once(root)
        else:
            # An open directory fd remains valid through rmdir; retain it so
            # the final mutation is still tied to the preflighted object.
            os.rmdir(path.name, dir_fd=parent_entry.handle)
            close_once(root)
        _flush_open_directory(parent_entry)
    finally:
        for entry in reversed(tuple(files.values())):
            close_once(entry)
        for entry in reversed(tuple(directories.values())):
            close_once(entry)
        for entry in reversed(chain):
            close_once(entry)


def publish_staging(
    staging: OwnedCorpusDirectory,
    target: Path,
) -> OwnedCorpusDirectory:
    if type(staging) is not OwnedCorpusDirectory:
        raise TypeError("publication requires an owned staging capability")
    parent = _absolute(target.parent)
    if staging.parent != parent or staging.path.parent != parent:
        raise LatentTrainingCorpusError(
            "atomic corpus publication requires one bounded parent"
        )
    parent_chain = _open_chain(parent)
    parent_entry = parent_chain[-1]
    target_owned = None
    try:
        if os.path.lexists(target):
            raise FileExistsError(target)
        with _open_owned(staging) as (_, held_staging, held_children):
            source_context = CorpusTreeSnapshot(staging.path)
            if not _same_object(
                source_context._directory_entries[""].identity,
                held_staging.identity,
            ):
                source_context.close()
                raise LatentTrainingCorpusError("owned staging pathname was replaced")
            for name, held_child in held_children.items():
                if not _same_object(
                    source_context._directory_entries[name].identity,
                    held_child.identity,
                ):
                    source_context.close()
                    raise LatentTrainingCorpusError(
                        "owned staging child was replaced"
                    )
        with source_context as source:
            source_files = {
                name: (value.size, value.sha256)
                for name, value in source.files.items()
            }
            source.assert_unchanged()

        # Promote the already-verified directory itself.  No target directory
        # exists before this atomic, no-replace operation, so there is no
        # create->open adoption window and no sensitive success-path copy.
        if os.name == "nt":
            promoter = _win_open(
                staging.path,
                directory=True,
                write=True,
                delete_access=True,
            )
            try:
                if not _same_object(promoter.identity, staging.identity):
                    raise LatentTrainingCorpusError(
                        "owned staging pathname was replaced"
                    )
                for name, identity in staging.child_identities:
                    child = _open_child(promoter, name, directory=True)
                    try:
                        if not _same_object(child.identity, identity):
                            raise LatentTrainingCorpusError(
                                "owned staging child was replaced"
                            )
                    finally:
                        _close_entry(child)
                _win_rename(promoter, parent_entry, target.name)
                target_owned = OwnedCorpusDirectory(
                    target,
                    parent,
                    target.name,
                    staging.identity,
                    staging.child_identities,
                )
                rebound = _win_open(
                    target,
                    directory=True,
                    share_delete=True,
                )
                try:
                    if not _same_object(rebound.identity, promoter.identity):
                        raise LatentTrainingCorpusError(
                            "atomic corpus promotion changed its target"
                        )
                finally:
                    _win_close(rebound.handle)
                if not _kernel32.FlushFileBuffers(promoter.handle):
                    raise _win_error("cannot flush promoted corpus directory")
                _flush_open_directory(parent_entry)
            finally:
                _win_close(promoter.handle)
        else:
            with _open_owned(staging) as (_, promoter, _):
                _posix_rename_noreplace(
                    parent_entry, staging.path.name, target.name
                )
                target_owned = OwnedCorpusDirectory(
                    target,
                    parent,
                    target.name,
                    staging.identity,
                    staging.child_identities,
                )
                _assert_named_object(parent_entry, target.name, promoter)
                _flush_open_directory(promoter)
                _flush_open_directory(parent_entry)

        if target_owned is None:
            raise AssertionError("publication did not capture target ownership")
        try:
            with CorpusTreeSnapshot(target) as published:
                if set(published.files) != set(source_files) or any(
                    (published.files[name].size, published.files[name].sha256)
                    != value
                    for name, value in source_files.items()
                ):
                    raise LatentTrainingCorpusError(
                        "published corpus differs from verified staging"
                    )
                published.assert_unchanged()
        except BaseException:
            remove_owned(target_owned)
            raise
    except BaseException:
        if target_owned is not None and os.path.lexists(target):
            remove_owned(target_owned)
        raise
    finally:
        for entry in reversed(parent_chain):
            _close_entry(entry)
    if os.path.lexists(staging.path):
        raise LatentTrainingCorpusError("verified staging was not atomically promoted")
    return target_owned


__all__ = [
    "CorpusTreeSnapshot",
    "OwnedCorpusDirectory",
    "owned_staging",
    "publish_staging",
    "remove_owned",
    "require_plain_parent",
    "write_new",
]
