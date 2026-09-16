# /audit.py

"""
integrity audit: completeness, labels, spacing, visual forensics.
Reference: Leclerc et al., IEEE TMI 38(9):2198-2210, 2019.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

ROOT = Path("CAMUS_dataset/database_nifti")
FIGURES_DIR = Path("figures/audit")
VIEWS, INSTANTS = ("2CH", "4CH"), ("ED", "ES")
LABELS_OK = {0, 1, 2, 3}      # Background / LV pool / Myocardium / LA
N_DEEP = 5

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load(path: Path):
    """Returns voxels in native storage dtype alongside in-plane spacing (mm)."""
    img = nib.load(path)
    return np.squeeze(np.asarray(img.dataobj)), img.header.get_zooms()[:2]


def name(patient: Path, view: str, instant: str, mask: bool = False) -> Path:
    """Returns canonical CAMUS file path for a (patient, view, instant) tuple."""
    suffix = "_gt" if mask else ""
    return patient / f"{patient.name}_{view}_{instant}{suffix}.nii.gz"


# Dataset Discovery & Integrity Validation
patients = sorted(p for p in ROOT.iterdir() if p.is_dir())
spacings = set()

for p in patients:
    for v in VIEWS:
        for i in INSTANTS:
            img_path = name(p, v, i, mask=False)
            mask_path = name(p, v, i, mask=True)

            if not img_path.exists():
                raise FileNotFoundError(f"Missing frame: {img_path}")
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing mask: {mask_path}")

            # Collect geometry spacing
            spacings.add(tuple(round(float(s), 3) for s in nib.load(img_path).header.get_zooms()[:2]))

            # Validate mask labels across ALL patients
            mask_data, _ = load(mask_path)
            labels = set(np.unique(mask_data).astype(int).tolist())
            assert labels <= LABELS_OK, f"{p.name} [{v}_{i}]: Unexpected labels {sorted(labels)}"

print(f"Validated {len(patients)} patients | {len(patients) * 4} frames | Spacings: {sorted(spacings)} mm")

# Visual Forensics & Intensity Auditing
for p in patients[:N_DEEP]:
    for v in VIEWS:
        image, _ = load(name(p, v, "ED"))
        mask_v, _ = load(name(p, v, "ED", mask=True))

        # Normalize float representation safely
        intensity = image.astype(np.float32) / max(image.max(), 1.0)

        # Overlay: Red=Intensity, Green=Mask, Blue=Zero
        overlay = np.dstack([intensity, (mask_v / 3.0) * 0.35, np.zeros_like(intensity)])

        plt.imsave(FIGURES_DIR / f"audit_{p.name}_{v}.png", overlay)
        plt.imsave(FIGURES_DIR / f"raw_{p.name}_{v}.png", intensity, cmap="gray")

        if v == "2CH":
            labels_found = np.unique(mask_v).astype(int).tolist()
            print(f"{p.name}: ED labels {labels_found} | Intensity range [{image.min()}, {image.max()}]")

print(f"Audit complete — inspect figures in {FIGURES_DIR}/")