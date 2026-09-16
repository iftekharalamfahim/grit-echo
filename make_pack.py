# make_pack.py

"""Assemble the upload pack (ED/ES only) with a SHA-256 manifest."""
import hashlib
import shutil
import zipfile
from pathlib import Path

SRC, WORK, OUT = Path("CAMUS_dataset"), Path("upload_v1"), Path("camus_v1.zip")


def keep(name: str) -> bool:
    """ED/ES images and masks, with per-view metadata."""
    if "half_sequence" in name:
        return False
    return name.endswith(".cfg") or name.endswith(".md") or "_ED" in name or "_ES" in name


def sha256(path: Path) -> str:
    """Streaming checksum, memory-bounded for multi-MB volumes."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# Clean up workspace
if WORK.exists():
    shutil.rmtree(WORK)

rows = []

# Process Patient NIfTI Volumes
for patient in sorted(p for p in (SRC / "database_nifti").iterdir() if p.is_dir()):
    dest_dir = WORK / "database_nifti" / patient.name
    dest_dir.mkdir(parents=True, exist_ok=True)

    for asset in sorted(patient.iterdir()):
        if keep(asset.name):
            target_path = dest_dir / asset.name
            shutil.copy2(asset, target_path)
            rows.append(
                (f"database_nifti/{patient.name}/{asset.name}", target_path.stat().st_size, sha256(target_path)))

# Process Official Dataset Splits
split_src = SRC / "database_split"
split_dest = WORK / "database_split"
if split_src.exists():
    shutil.copytree(split_src, split_dest, dirs_exist_ok=True)
    for asset in sorted(split_dest.iterdir()):
        if asset.is_file():
            rows.append((f"database_split/{asset.name}", asset.stat().st_size, sha256(asset)))

# Write Manifest CSV
manifest_path = WORK / "MANIFEST.csv"
manifest_path.write_text("path,bytes,sha256\n" + "".join(f"{p},{b},{s}\n" for p, b, s in rows))

# Create Zip Pack (POSIX compliant)
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_STORED) as archive:
    for file in sorted(WORK.rglob("*")):
        if file.is_file():
            archive.write(file, file.relative_to(WORK).as_posix())

# Optional Cleanup of Working Directory
shutil.rmtree(WORK)

print(f"Pack assembled: {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, {len(rows) + 1} files including MANIFEST.csv)")