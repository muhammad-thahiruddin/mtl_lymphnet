### 📄 Dataset Specifications

* **Subject:** Health Sciences, Medical Sciences & Pharmacology
* **Specific Subject Area:** Oncology, Radiology
* **Type of Data:** Images and Excel Spreadsheets

### ⚙️ Data Collection & Source

* **Methodology:** Retrospective data extraction from electronic medical records and radiological PACS systems. The study evaluated patients with papillary thyroid carcinoma and suspected cervical lymph node metastasis across two independent clinical centers.
* **Institution:** Shahid Beheshti University of Medical Sciences, Tehran, Iran

### 🔗 Data Accessibility

* **Repository Name:** QAMEBI
* **Direct URL:** [https://qamebi.com/cervical-lymph-node](https://qamebi.com/cervical-lymph-node)
* **File Details:** Data is organized into `Center 1.zip` and `Center 2.zip` folders. These contain ultrasound images with segmentation masks, fusion images, and comprehensive imaging features in Excel format.

### 📚 Primary Research Articles

* **Dataset Descriptor:** Mohammadi A, Mohebbi A, Mirza-Aghazadeh-Attari M, et al. *"LymphUs: A multicenter open-access database of lymph node ultrasound images in patients with papillary thyroid carcinoma for clinical and artificial intelligence research."* **Data in Brief**. 2026;66:112694. DOI: [10.1016/j.dib.2026.112694](https://doi.org/10.1016/j.dib.2026.112694)

* **Related Comparative Study:** Abbasian Ardakani A, Mohammadi A, Mirza-Aghazadeh-Attari M, et al. *"Diagnosis of Metastatic Lymph Nodes in Patients With Papillary Thyroid Cancer: A Comparative Multi-Center Study of Semantic Features and Deep Learning-Based Models."* **Journal of Ultrasound in Medicine**. 2023;42(6):1211–1221. DOI: [10.1002/jum.16131](https://doi.org/10.1002/jum.16131)

### 📁 Folder Structure Required for This Framework

This repository does **not** include the raw image files under `Center 1/`
and `Center 2/` (excluded via `.gitignore`, both to respect the original
dataset's own distribution terms via QAMEBI and to keep this repository
small). To run any experiment locally, download `Center 1.zip` and
`Center 2.zip` from the QAMEBI link above and arrange them under `dataset/`
exactly as follows:

```
dataset/
    about_dataset.md          (this file)
    Center 1/
        Benign/
            1 Benign Image.bmp
            1 Benign Mask.tif
            1 Benign Lesion.bmp
            2 Benign Image.bmp
            ...
        Malignant/
            1 Malignant Image.bmp
            1 Malignant Mask.tif
            1 Malignant Lesion.bmp
            ...
    Center 2/
        Benign/
            ... (same three-file-per-sample pattern)
        Malignant/
            ...
```

Each sample consists of exactly three files sharing a numeric ID and class
name:

| Suffix | Format | Meaning |
|---|---|---|
| ` Image.bmp` | BMP | Original B-mode ultrasound image |
| ` Mask.tif` | TIF | Binary segmentation mask (lymph node boundary) |
| ` Lesion.bmp` | BMP | Pre-cropped lesion patch |

Numeric IDs restart at 1 within each `Center/Class` combination (e.g. both
`Center 1/Benign/1 Benign Image.bmp` and `Center 2/Malignant/1 Malignant
Image.bmp` exist -- they are different samples, disambiguated by their full
path, not by ID alone).

`outputs_v2/dataset_inventory.csv` (already included in this repository) was
built by scanning this exact folder structure and lists every sample's path
and class without requiring the underlying images to be present, so you can
inspect the expected layout even before downloading the raw data.
