# Cross-Device Human Activity Recognition with Smartphone Accelerometers

Does a human activity recognition (HAR) model trained on one smartphone still work on a different smartphone model? This project investigates that question using the **Heterogeneity Human Activity Recognition (HHAR)** dataset, comparing **within-device** and **cross-device** performance across three classical machine learning models.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter%20%2F%20Colab-orange)

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
4. [Results](#results)
5. [Limitations](#limitations)
6. [Repository Structure](#repository-structure)
7. [Getting Started](#getting-started)
8. [Reproducibility](#reproducibility)
9. [Citation](#citation)
10. [Author](#author)

---

## Overview

Smartphones from different manufacturers use different accelerometer hardware and sampling rates. A model that performs well on the phone it was trained on can lose accuracy on a phone it has never seen. This project measures that gap.

**Research question:** How much does activity recognition performance change when a model is tested on a smartphone model that was not used for training?

**What the notebook does:**

- Loads and cleans raw accelerometer data (about 13 million labeled readings)
- Segments the signal into 3-second windows and extracts statistical features
- Evaluates **Logistic Regression**, **Random Forest**, and **XGBoost**
- Compares **within-device** and **cross-device** settings on unseen users
- Repeats experiments over 5 random seeds and runs paired Wilcoxon signed-rank tests

---

## Dataset

**Source:** [Heterogeneity Activity Recognition (UCI Machine Learning Repository)](https://archive.ics.uci.edu/dataset/344/heterogeneity+activity+recognition), file `Phones_accelerometer.csv`.

| Property | Value |
|---|---|
| Sensor | Smartphone accelerometer (x, y, z) |
| Users | 9 (labeled `a` to `i`) |
| Phone models | 4: `nexus4`, `s3`, `s3mini`, `samsungold` |
| Physical devices | 8 (two units per model) |
| Activities | 6: `bike`, `sit`, `stand`, `walk`, `stairsup`, `stairsdown` |
| Labeled readings | 11,279,275 (after removing 1,783,200 rows labeled `null`) |

**Device heterogeneity.** The phones record at noticeably different rates. The median time between consecutive readings was about **5 ms** on the Nexus 4, **10 ms** on the S3 and S3 Mini, and **20 ms** on the older Samsung device. This is one of the main sources of the cross-device gap.

> The dataset is **not included** in this repository because of its size. Download it from the link above (see [Getting Started](#getting-started)).

---

## Methodology

### 1. Preprocessing

- Removed rows with no activity label (`null`)
- Sorted readings by user, phone model, device, and timestamp
- Started a new segment whenever the activity changed or the gap between readings exceeded 1 second

### 2. Windowing and feature extraction

- Non-overlapping **3-second windows** inside each segment
- Kept only windows with duration of at least 2.5 s and at least 100 samples
- **16 features per window**: mean, standard deviation, minimum, and maximum for each of `x`, `y`, `z`, and the acceleration magnitude

Result: **33,075 windows** across 6 activities, with no missing values.

### 3. Experimental design

Two evaluation setups were used.

**Preliminary: leave-one-device-model-out.** Train on three phone models, test on the fourth. Train and test data come from the same users.

**Main experiment: user-independent within vs cross-device.** Users are split into fixed groups: train users `a` to `g`, test users `h` and `i`. For each target phone model:

| Condition | Training data | Test data |
|---|---|---|
| **Within-Device** | Same phone model, users `a` to `g` | Same phone model, users `h`, `i` |
| **Cross-Device** | Other three phone models, users `a` to `g` | Target phone model, users `h`, `i` |

To make the comparison fair, the cross-device training set is **sampled to match the size and per-activity class counts** of the within-device training set. Experiments are repeated with seeds `42, 123, 2024, 7, 99`.

**Models:** Logistic Regression (with standard scaling), Random Forest (200 trees), XGBoost (200 trees, depth 6).

**Metric:** Macro-F1, which weights all six activities equally.

---

## Results

### Preliminary experiment (leave-one-device-model-out, users shared)

| Model | Mean Accuracy | Mean Macro-F1 |
|---|---|---|
| Logistic Regression | 0.817 | 0.813 |
| Random Forest | 0.902 | 0.901 |
| XGBoost | 0.908 | 0.907 |

These scores are high, but training and test data contain the same people. Once users are separated (below), scores drop substantially, which shows that **user-level leakage inflates the results** of this setup.

### Main experiment (user-independent, size-matched, 5 seeds)

Mean Macro-F1 (%) per target device. "Change" is cross-device minus within-device, in percentage points.

| Model | Target device | Within-Device | Cross-Device | Change (pp) |
|---|---|---:|---:|---:|
| Logistic Regression | nexus4 | 64.86 | 72.32 | +7.46 |
| Logistic Regression | s3 | 70.13 | 77.74 | +7.61 |
| Logistic Regression | s3mini | 80.62 | 77.65 | -2.97 |
| Logistic Regression | samsungold | 78.63 | 73.95 | -4.68 |
| Random Forest | nexus4 | 72.51 | 68.54 | -3.97 |
| Random Forest | s3 | 69.25 | 63.53 | -5.72 |
| Random Forest | s3mini | 73.24 | 72.90 | -0.34 |
| Random Forest | samsungold | 71.61 | 69.91 | -1.70 |
| XGBoost | nexus4 | 71.63 | 68.68 | -2.95 |
| XGBoost | s3 | 71.56 | 62.97 | -8.59 |
| XGBoost | s3mini | 81.18 | 74.81 | -6.37 |
| XGBoost | samsungold | 71.33 | 74.67 | +3.34 |

**Averaged over the four target devices:**

| Model | Within-Device | Cross-Device | Change (pp) |
|---|---:|---:|---:|
| Logistic Regression | 73.6 | 75.4 | +1.9 |
| Random Forest | 71.7 | 68.7 | -2.9 |
| XGBoost | 73.9 | 70.3 | -3.6 |

### Figures

<p align="center">
  <img src="images/Figure_1_Within_vs_Cross_MacroF1.png" width="48%" alt="Within-device vs cross-device Macro-F1 by model">
  <img src="images/Figure_2_Cross_Device_Device_Comparison.png" width="48%" alt="Cross-device Macro-F1 by target device">
</p>

### Key findings

- **Moving to an unseen phone model usually hurts the tree-based models.** Random Forest and XGBoost lost about 3 to 4 percentage points of Macro-F1 on average, with the largest drops on the Galaxy S3 (up to 8.6 pp for XGBoost).
- **The effect depends on the device.** Not every phone is hard to generalize to. The older Samsung device and the Nexus 4 sometimes improved or barely changed, depending on the model.
- **Logistic Regression behaved differently.** It gained on two targets and lost on two, with a small positive average. Its within-device scores were also the lowest on some devices, so the model may be limited by its capacity rather than helped by cross-device data.
- **Statistical testing was inconclusive.** A paired Wilcoxon signed-rank test over 5 seeds cannot produce a two-sided p-value below 0.0625, and most comparisons sit at that floor. The results show a consistent direction, not statistical significance.

---

## Limitations

- **Only 9 users.** The test group is just two people (`h`, `i`), so results may change with a different user split. Seeds vary the sampling and training randomness but **not** the user split.
- **Accelerometer only.** The gyroscope and smartwatch data in the original dataset are not used.
- **Hand-crafted statistical features.** No frequency-domain features and no deep learning models were tested.
- **Non-overlapping windows** of a fixed 3-second length; other window sizes were not explored.
- **Raw data are not resampled** to a common rate, so sampling-rate differences remain a confounder.

---

## Repository Structure

```
.
├── HAR.ipynb          # Full pipeline: preprocessing, features, experiments, figures
├── README.md
├── requirements.txt
└── images/            # Result figures used in this README
```

---

## Getting Started

The notebook was developed in **Google Colab** and reads and writes files on Google Drive.

1. **Download the dataset** from the [UCI repository](https://archive.ics.uci.edu/dataset/344/heterogeneity+activity+recognition). You will get `heterogeneity+activity+recognition.zip`.
2. **Open `HAR.ipynb` in Google Colab** (File, then Open notebook, then GitHub tab).
3. **Run the first cells**, which mount Google Drive and prompt you to upload the zip file. The notebook copies it to `MyDrive/ML_Project/` and extracts the data.
4. **Run the remaining cells in order.** Intermediate Parquet files and result CSVs are saved to the same `ML_Project` folder.

**To run locally instead**, install the dependencies, remove the Colab-specific cells (`google.colab` imports), and update the file paths that point to `/content/drive/MyDrive/ML_Project/`.

```bash
pip install -r requirements.txt
jupyter notebook HAR.ipynb
```

> Note: the raw accelerometer file is large (several GB when extracted), so loading it needs sufficient RAM or chunked reading, which the notebook uses where needed.

---

## Reproducibility

- All models use fixed random seeds (`42`, plus `123, 2024, 7, 99` for the multi-seed runs).
- The user split is fixed: train `a` to `g`, test `h` and `i`.
- Results are written to CSV files by the notebook, so tables in this README can be regenerated.

---

## Citation

If you use this work, please also cite the original dataset:

> Stisen, A., Blunck, H., Bhattacharya, S., Prentow, T. S., Kjærgaard, M. B., Dey, A., Sonne, T., & Jensen, M. M. (2015). *Smart Devices are Different: Assessing and Mitigating Mobile Sensing Heterogeneities for Activity Recognition.* Proceedings of the 13th ACM Conference on Embedded Networked Sensor Systems (SenSys).

---

## Author

**Your Name**
[GitHub](https://github.com/feritbulut) | [LinkedIn](https://www.linkedin.com/in/feritbulut)

---

## License

This project is released under the MIT License. See `LICENSE` for details. The HHAR dataset is subject to its own license terms from the UCI Machine Learning Repository.