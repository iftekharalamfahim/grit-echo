"""
Build the CAMUS half-sequence companion pack for private Kaggle upload.

Creates a flat, store-only zip of every *_half_sequence.nii.gz volume so
Stage 2 (granule tracking) can run frame-wise inference on Kaggle without
rebuilding the Stage 0 ED/ES pack.
"""
import zipfile
from pathlib import Path

SRC = Path("CAMUS_dataset/database_nifti")
OUT = Path("camus_sequences.zip")
EXPECTED = 1000  # 500 patients x 2 views


def main() -> None:
    files = sorted(SRC.glob("*/*half_sequence.nii.gz"))
    if len(files) != EXPECTED:
        raise SystemExit(f"expected {EXPECTED} half-sequence volumes, found {len(files)}")
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_STORED) as zf:
        for f in files:
            # flat layout: patient0001_2CH_half_sequence.nii.gz
            zf.write(f, arcname=f.name)
    print(f"wrote {OUT.name}: {len(files)} files, {OUT.stat().st_size / 1_000_000:.1f} MB")


if __name__ == "__main__":
    main()