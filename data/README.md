# Dataset Preparation Guide

This project is trained and evaluated using a hybrid collection of cervical cytology images categorized according to the **Bethesda System**:
- **NILM**: Negative for Intraepithelial Lesion or Malignancy (Normal / Benign)
- **LSIL**: Low-grade Squamous Intraepithelial Lesion (Mild dysplasia, CIN-1)
- **HSIL**: High-grade Squamous Intraepithelial Lesion (Moderate/Severe dysplasia, CIN-2 / CIN-3, CIS)

---

## 1. Public Datasets Sources

1. **SIPaKMeD**:
   - Reference: Plissiti et al., *SIPAKMED: A new dataset for feature and image based classification of normal and pathological cervical cells in Pap smear images*, IEEE ICIP 2018.
   - Contains isolated cells and clusters across 5 categories, mapped into NILM, LSIL, and HSIL.
2. **Mendeley Liquid-Based Cytology (LBC)**:
   - Reference: Hussain, *Cervical Cytology Dataset*, Mendeley Data, 2019.
   - Contains high-resolution micrographs categorized by dysplasia grade.

---

## 2. Directory Layout for Training

Place your raw or structured images inside a folder (default `data/` or any local directory specified via `--data_dir`):

```text
data/
├── HSIL/
│   ├── cell_001.png
│   ├── cell_002.jpg
│   └── ...
├── LSIL/
│   ├── cell_101.png
│   ├── cell_102.jpg
│   └── ...
└── NILM/
    ├── cell_201.png
    ├── cell_202.jpg
    └── ...
```

When running `python train.py`, the pipeline automatically performs a stratified, reproducible split (`70% train`, `15% validation`, `15% test`) and applies the tissue-aware preprocessing and Reinhard color normalization pipeline.
