"""Path resolution helpers.

Data paths (values coming from the competition CSVs) are resolved relative to a
configurable ``DATA_DIR``. Artifact paths (checkpoints, predictions, submission)
are resolved relative to the repository root so they are portable between
machines regardless of where the data happens to live.
"""
from pathlib import Path

# TODO: set to the folder that contains train/, test_stage1/ and (optionally) nanobanana/
DATA_DIR = Path('')

# repo root: this file lives at <repo>/src/paths.py
SCRIPT_DIR = Path(__file__).resolve().parent.parent


def _normalize(path_str: str) -> str:
    return path_str.strip().replace('\\', '/')


def resolve_path(path: str) -> Path:
    """Data paths — relative to DATA_DIR (absolute paths are returned as-is)."""
    p = Path(_normalize(path))
    return p if p.is_absolute() else DATA_DIR / p


def resolve_output_path(path: str) -> Path:
    """Artifact paths (checkpoints/predictions/submission) — relative to the repo root."""
    p = Path(_normalize(path))
    return p if p.is_absolute() else SCRIPT_DIR / p


def resolve_data_path(csv_relpath: str, base_dir: Path) -> Path:
    """Resolve a path value taken from one of the competition CSVs, ALWAYS
    underneath ``base_dir``.

    The provided CSVs bake in absolute paths from whichever machine wrote them
    (``train.csv`` stores e.g. ``D:\\\\stage1/train/img/x.jpg``) and, on top of
    that, a leading ``stage1`` segment that matches no real directory. Both are
    properties of the CSV content, not of how a given copy was unzipped.

    The absolute prefix is the dangerous one: ``Path('C:/dest') / Path('D:/src')``
    evaluates to ``D:/src`` in Python, so a naive join silently ignores
    ``base_dir`` and sends every read back to the drive named in the CSV --
    making it impossible to relocate the dataset (e.g. HDD -> SSD) and doing so
    without any error to notice.

    So: drop the anchor if the value is absolute, then try progressively longer
    suffixes under ``base_dir`` until one exists. That absorbs the ``stage1``
    prefix and any similar leading junk without hardcoding its name.
    """
    rel = _normalize(csv_relpath)
    rel_path = Path(rel)
    parts = rel_path.parts[1:] if rel_path.is_absolute() else rel_path.parts
    if not parts:
        raise FileNotFoundError(f"Empty data path value: {csv_relpath!r}")

    tried = []
    for start in range(len(parts)):
        candidate = base_dir / Path(*parts[start:])
        if candidate.exists():
            return candidate
        tried.append(str(candidate))
    raise FileNotFoundError(
        f"Could not resolve '{csv_relpath}' under {base_dir}; tried: {tried}"
    )
