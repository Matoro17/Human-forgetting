```markdown
# Dataset: – Renal Lesion Classification - Kfold Organization

This folder contains the dataset used in the experiments for renal lesion classification using self-supervised and supervised learning methods for the LACaD Lab.  
It is organized to support **k-fold cross-validation**, with optional **data augmentation** and **class exclusion**.

The directory contains four CSV files used to define different dataset configurations for the experiments:

- **kfold_augmentations_clear_classes.csv** — This file excludes the classes `2_Esclerose_Pura_Sem_Crescente` and `4_Hipercelularidade_Pura_Sem_Crescente` from the dataset.  
- **kfold_augmentations.csv** — This version includes all classes and contains duplicated indexes for images with augmentation flags, indicating the type of data augmentation applied in each experiment.  
- **kfold_symlinks_Normal_oversample.csv** — This file provides a balanced version of the dataset by duplicating samples from underrepresented classes, without applying augmentation transformations.  
- **kfold_symlinks.csv** — This serves as the baseline dataset split, defining the distribution of samples across the 5 folds used during cross-validation.


### Class Distribution in the FIOCRUZ Histopathological Image Dataset

| Classe                              | Total | AZAN | HE   | PAMS | PAS  | PSI |
|------------------------------------|:------:|:----:|:----:|:----:|:----:|:---:|
| **Total de Amostras**              | **12524** | **1171** | **5959** | **1254** | **3537** | **7** |
| Normal                             | 2695  | 223  | 1585 | 345  | 542  | 0 |
| Amiloidose                         | 374   | 31   | 145  | 96   | 102  | 0 |
| Esclerose Pura Sem Crescente       | 1481  | 233  | 672  | 104  | 472  | 0 |
| Hipercelularidade                  | 3134  | 257  | 1890 | 0    | 987  | 0 |
| Hipercelularidade Pura Sem Crescente | 224 | 60   | 0    | 0    | 164  | 0 |
| Crescent                           | 1104  | 121  | 467  | 157  | 359  | 0 |
| Membranous                         | 1539  | 136  | 712  | 324  | 367  | 0 |
| Sclerosis                          | 617   | 0    | 276  | 122  | 219  | 0 |
| Podocytopathy                      | 505   | 90   | 65   | 106  | 244  | 0 |
| Acúmulo de Neutrófilos             | 713   | 0    | 486  | 0    | 175  | 52 |
| Depósitos Hialinos                 | 94    | 14   | 51   | 0    | 29   | 0 |
| Necrose Fibrinoide                 | 34    | 6    | 20   | 0    | 8    | 0 |

---

## 📁 Directory Structure

```

dataset-mestrado-Gabriel/
│
├── 0_Amiloidose/
├── 1_Normal/
├── 2_Esclerose_Pura_Sem_Crescente/
├── 3_Hipercelularidade/
├── 4_Hipercelularidade_Pura_Sem_Crescente/
├── 5_Crescent/
├── 6_Membranous/
├── 7_Sclerosis/
├── 8_Podocytopathy/
├── 9_acumulo_neutrofilos/
├── 10_depositos_hialinos/
├── 11_necrose_fibrinoide/
│
├── kfold_augmentations.csv
├── kfold_augmentations_clear_classes.csv
└── (optional generated files)

````

Each folder corresponds to a **histopathological lesion class**.  
Inside each class directory, there are **subdirectories** (e.g., `HE/`, `PAMS/`, `AZAN/`) containing the actual image files.

---

## 🧬 Classes

| ID | Folder Name | Description (Lesion Type) |
|----|--------------|----------------------------|
| 0 | 0_Amiloidose | Amyloidosis |
| 1 | 1_Normal | Normal glomerulus |
| 2 | 2_Esclerose_Pura_Sem_Crescente | Pure sclerosis without crescent :x: |
| 3 | 3_Hipercelularidade | Hypercellularity |
| 4 | 4_Hipercelularidade_Pura_Sem_Crescente | Pure hypercellularity without crescent |
| 5 | 5_Crescent | Crescent formation |
| 6 | 6_Membranous | Membranous glomerulopathy |
| 7 | 7_Sclerosis | General sclerosis |
| 8 | 8_Podocytopathy | Podocytopathy |
| 9 | 9_acumulo_neutrofilos | Neutrophil accumulation |
| 10 | 10_depositos_hialinos | Hyaline deposits |
| 11 | 11_necrose_fibrinoide | Fibrinoid necrosis |

---

:x: IMPORTANT DETAIL -> The 2_Esclerose_Pura_Sem_Crescente class was alerted to not be that actual class, so avoid usage.

## 📄 CSV Files

### 1. `kfold_augmentations_clear_classes.csv`

Contains the k-fold split information without augmented samples.

**Structure:**
```csv
fold,split,class_name,image_path
0,train,6_Membranous,6_Membranous/HE/6_Membranous_HE_113.jpeg
0,train,11_necrose_fibrinoide,11_necrose_fibrinoide/AZAN/2022PC0202 (65).JPG
0,train,5_Crescent,5_Crescent/PAMS/5_Crescent_PAMS_40.jpeg
````

| Column         | Description                                   |
| -------------- | --------------------------------------------- |
| **fold**       | Fold index (0 to N-1)                         |
| **split**      | Data split: `train` or `test`                 |
| **class_name** | Folder name corresponding to the lesion class |
| **image_path** | Relative path from dataset root               |

---

### 2. `kfold_augmentations.csv`

Contains both the k-fold split and the **augmentation information**.

**Structure:**

```csv
fold,split,class_name,image_path,augmentation_type
0,train,11_necrose_fibrinoide,11_necrose_fibrinoide/HE/2022PC0238-IMG_4926.JPG,jitter
0,train,1_Normal,1_Normal/HE/1_Normal_HE_752.jpeg,original
0,train,0_Amiloidose,0_Amiloidose/PAMS/0_Amiloidose_PAMS_88.jpg,blur
```

| Column                | Description                                                               |
| --------------------- | ------------------------------------------------------------------------- |
| **fold**              | Fold index (0 to N-1)                                                     |
| **split**             | Data split: `train` or `test`                                             |
| **class_name**        | Folder name corresponding to the lesion class                             |
| **image_path**        | Relative path from dataset root                                           |
| **augmentation_type** | Type of augmentation applied: `original`, `jitter`, `blur`, or `solarize` |

---

## ⚙️ CSV Generation Script

**Script:** `create_kfold_csv.py`

This script generates a stratified k-fold split of the dataset and saves it as a CSV file.
It ensures that:

* Each image appears **exactly once in the test set** across folds.
* Optional **oversampling** balances the class distribution in training.
* Optionally **ignores specific classes** if desired.

### **Usage Example**

```bash
python create_kfold_csv.py /path/to/dataset-mestrado-Gabriel \
  --folds 5 \
  --output_csv /path/to/output/kfold_augmentations_clear_classes.csv
```

**Optional arguments:**

* `--ignore_classes` → list of class names to exclude
* `--oversample` → enables oversampling of minority classes

---

## 🧠 Dataset Loader

**Class:** `CustomDatasetFromCSV`

This class loads images using the CSV files generated above, compatible with **PyTorch** `DataLoader`.

### **Initialization Example**

```python
from datasets.custom_dataset_csv import CustomDatasetFromCSV
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = CustomDatasetFromCSV(
    csv_file="kfold_augmentations_clear_classes.csv",
    fold=0,
    split="train",
    data_dir="/path/to/dataset-mestrado-Gabriel",
    transform=transform
)
```

### **Key Features**

* Supports **multi-fold** training and testing.
* Allows **excluding classes** dynamically.
* Handles **binary classification** setups with `positive_classes`.
* Validates paths and images automatically.

---

## 🧩 Typical Workflow

1. **Organize dataset** under the described folder structure.
2. **Run** `create_kfold_csv.py` to generate the CSV splits.
3. **Use** `CustomDatasetFromCSV` to load data in PyTorch for training.
4. **(Optional)** Generate augmented CSVs using your augmentation pipeline.

---

## 🗂️ File Summary

| File                                    | Purpose                          |
| --------------------------------------- | -------------------------------- |
| `create_kfold_csv.py`                   | Generates k-fold split CSVs      |
| `kfold_augmentations_clear_classes.csv` | Base CSV (without augmentations) |
| `kfold_augmentations.csv`               | Includes augmentations           |
| `datasets/custom_dataset_csv.py`        | PyTorch dataset loader class     |

---

## ✅ Notes

* The dataset is intended for research on **renal glomerular lesion classification**.
* Each CSV is **self-contained** — image paths are relative to the dataset root.
* Before running the training pipeline, verify that `data_dir` in your scripts points to the dataset folder.

```


