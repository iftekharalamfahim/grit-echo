<div align="center">

# GRIT-Echo

**GR**anular **I**nformation & **R**einforcement **T**racking in **Echo**cardiography

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

*Granular-computing-based reliability gating for left-ventricle segmentation and LVEF estimation, coupled with a reinforcement-learning contour-tracking framework on the CAMUS dataset.*

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Research Questions](#research-questions)
- [Pipeline Architecture](#pipeline-architecture)
- [Repository Structure](#repository-structure)
- [Environment & System Requirements](#environment--system-requirements)
- [Results](#results)
- [Reproduction Steps](#reproduction-steps)
- [Roadmap & Progress](#roadmap--progress)
- [Dataset & Mandatory Citation](#dataset--mandatory-citation)
- [Citing This Work](#citing-this-work)
- [License](#license)

---

## Overview

Automated left-ventricle (LV) segmentation in 2D echocardiography achieves high mean accuracy while leaving per-case reliability unquantified: traditional networks emit point predictions even where acoustic shadowing and speckle noise leave the boundary undefined. **GRIT-Echo** addresses this across two integrated stages:

- **Stage 1 (In Progress):** A supervised U-Net baseline with a ResNet-34 encoder, trained and evaluated on official CAMUS splits and validated against the clinical endpoint (LVEF via Simpson's biplane method). A granular-computing layer converts per-pixel predictive entropy into information granules of boundary confidence (crisp tissue / vague boundary / artifact), enabling automatic reliability gating.
- **Stage 2 (Planned):** A deep reinforcement-learning agent (DQN, discrete action space) that refines and tracks the LV contour across the cardiac cycle, consuming the granular uncertainty map as a state channel.

---

## Research Questions

- **RQ1:** Does predictive entropy concentrate predictably at anatomical boundaries and artifact regions, particularly within the expert-rated poor-quality subgroup?
- **RQ2:** Do granule-level statistics reliably predict per-case Dice and LVEF error — enabling unreliable cases to be flagged automatically before clinical deployment?

---

## Pipeline Architecture
   ![GRIT-Echo technical architecture](figures/Grit_Echo_pipeline_architecture.png)

---

## Repository Structure

```
grit-echo/
├── audit.py            # Stage 0: Dataset integrity audit + forensic figures
├── make_pack.py        # Builds the license-compliant Kaggle upload pack
├── figures/            # Audit overlays and raw-frame forensics
├── notebooks/          # Thin drivers; full experiments run on Kaggle
├── grit_echo/          # Core package: data, model, metrics, granules, rl
├── reports/            # Project reports, one-page summaries, slides
└── .gitignore          # Data, environments, and archives excluded by design
```

---

## Environment & System Requirements

| Component | Requirement / Specification |
|:---|:---|
| **Python** | 3.10+ |
| **Frameworks** | PyTorch 2.x, Torchvision, Gymnasium |
| **Medical IO** | `nibabel`, `SimpleITK`, `opencv-python` |
| **Execution Context** | Kaggle Notebooks (NVIDIA T4 ×2) / Local CUDA GPU |

---
## Results

All numbers below are reproducible from the cited Kaggle notebook version. 

---

## Reproduction Steps

1. **Data Acquisition:** Request the official CAMUS dataset from the [Human Heart Project](https://humanheart-project.creatis.insa-lyon.fr/database/). *This repository does not redistribute raw data.*
2. **Local Environment Setup:**
   ```bash
   pip install nibabel numpy matplotlib opencv-python
   python audit.py
   ```
3. **Dataset Packing:** 
    - **Stage 1 sequences:** Run `python make_pack.py` to generate the compressed ED/ES pack (~300 MB) for private Kaggle execution.
    - **Stage 2 sequences:** run `python make_seq_pack.py` (1,000 half-sequence volumes, ~3.4 GB) and upload the zip as the private Kaggle dataset `camus-half-sequences`.
4. **Experimental Runs:** Execute via the Kaggle Driver (`NVIDIA T4 ×2`, `SEED=42`).

---


## Dataset & Mandatory Citation

S. Leclerc, E. Smistad, J. Pedrosa, A. Ostvik, et al., "Deep Learning for Segmentation using an Open Large-Scale Dataset in 2D Echocardiography," *IEEE Transactions on Medical Imaging*, 38(9):2198-2210, 2019. DOI: [10.1109/TMI.2019.2900516](https://doi.org/10.1109/TMI.2019.2900516).

*We thank CREATIS and the Human Heart Project for open data access.*

---

## Citing This Work

```bibtex
@misc{grit_echo_2026,
  title        = {GRIT-Echo: Granular Information \& Reinforcement Tracking for Echocardiography},
  author       = {Fahim, Iftekhar Alam},
  year         = {2026},
  howpublished = {\url{https://github.com/iftekharalamfahim/grit-echo}}
}
```

---

## License

Code is licensed under the [MIT License](LICENSE). The CAMUS dataset remains subject to its original license terms.