"""File helpers shared by the pipeline scripts."""
import fcntl
import json
import os
from pathlib import Path


def write_atomic(path, text):
    """Write via a temp file + rename, so readers (and parallel runs) never see a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def save_npz(path, **arrays):
    """np.savez via a temp file + rename, so a concurrent reader never loads a half-written cache."""
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as f:      # a file object, so np.savez doesn't append another .npz
        np.savez(f, **arrays)
    os.replace(tmp, path)


def write_json(path, obj, **kw):
    write_atomic(path, json.dumps(obj, **kw))


def locked_update(path, fn, **kw):
    """Read-modify-write a JSON file under an flock (one run per GPU may share it). fn mutates the dict."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name(f".{path.stem}.lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(path.read_text()) if path.exists() else {}
        fn(data)
        write_json(path, data, **kw)
