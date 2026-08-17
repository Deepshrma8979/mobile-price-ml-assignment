# Mobile Price Classification — ML Assignment 2

**Name:** Deepanshi Sharma
**BITS ID:** 2025ac05642

## a. Problem Statement

Classify a mobile phone into one of four price range categories (0 = low cost,
1 = medium cost, 2 = high cost, 3 = very high cost) based on its hardware
specifications (RAM, battery power, camera resolution, screen dimensions,
processor cores, etc.). Five classification models are trained on the same
dataset, compared across six evaluation metrics, and served through a
Streamlit web app.

## b. Dataset Description

| S.No. | Property | Value |
|---|---|---|
| 1 | Source | Mobile Price Classification, Kaggle (iabhishekofficial) |
| 2 | File used | train.csv |
| 3 | Instances | 2000 |
| 4 | Features | 20 |
| 5 | Target | price_range (4 classes) |
| 6 | Task | Multi-class classification |

Features (20): battery_power, blue, clock_speed, dual_sim, fc, four_g,
int_memory, m_dep, mobile_wt, n_cores, pc, px_height, px_width, ram, sc_h,
sc_w, talk_time, three_g, touch_screen, wifi — all numeric, no categorical
encoding required.

Class distribution: perfectly balanced — 500 instances in each of the 4
classes (0, 1, 2, 3). No missing values in the raw data.

Preprocessing: missing values imputed with the column median (precautionary,
none were present), features standardised with StandardScaler, 80/20
stratified train-test split with random_state=42.

## c. GitHub Repository Link

https://github.com/Deepshrma8979/mobile-price-ml-assignment

## d. Models Used

| S.No. | Model | Parameters |
|---|---|---|
| 1 | Logistic Regression | max_iter=1000, random_state=42 |
| 2 | Decision Tree | random_state=42 |
| 3 | kNN | default (n_neighbors=5) |
| 4 | Naive Bayes | Gaussian |
| 5 | Random Forest (Ensemble) | random_state=42, n_estimators=100 (default) |

Gaussian Naive Bayes was used rather than Multinomial because all features
are continuous and standardised.

### Comparison of Evaluation Metrics

AUC is one-vs-rest (weighted); Precision, Recall and F1 are weighted averages.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9650 | 0.9987 | 0.9650 | 0.9650 | 0.9650 | 0.9534 |
| Decision Tree | 0.8300 | 0.8867 | 0.8319 | 0.8300 | 0.8302 | 0.7738 |
| kNN | 0.5000 | 0.7697 | 0.5211 | 0.5000 | 0.5054 | 0.3350 |
| Naive Bayes | 0.8100 | 0.9506 | 0.8113 | 0.8100 | 0.8105 | 0.7468 |
| Random Forest (Ensemble) | 0.8800 | 0.9767 | 0.8796 | 0.8800 | 0.8797 | 0.8400 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best model overall on every metric (Accuracy 0.965, AUC 0.9987, MCC 0.9534). The four price tiers are largely determined by a near-linear relationship with a small number of dominant, monotonic features (chiefly `ram`, and to a lesser extent `battery_power`). Once these are standardised, a linear decision boundary separates the ordered price bands almost perfectly, which is why the simplest model wins here — this is not overfitting, since the same pattern held cleanly on the untouched 20% test split. |
| Decision Tree | Clearly behind both Logistic Regression and Random Forest (Accuracy 0.830, AUC 0.887 — the lowest AUC among non-kNN models). A single tree makes axis-aligned, greedy splits and has no mechanism to represent a smooth linear combination of `ram` and `battery_power`; it also overfits noise in the training split. Its leaf-count-based probability estimates are coarse, which further drags down AUC even where the predicted class label is correct. |
| kNN | The weakest model by a wide margin (Accuracy 0.500, MCC 0.335). Despite scaling, Euclidean distance across all 20 features gets diluted by several near-binary, largely irrelevant dimensions (`blue`, `dual_sim`, `three_g`, `four_g`, `touch_screen`, `wifi`) that don't distinguish price tiers. This is a textbook curse-of-dimensionality effect — the "nearest" neighbours in 20-D space are often not the phones most similar in price-driving specs. |
| Naive Bayes | Middling accuracy (0.810) but a notably strong AUC (0.9506) — meaning it usually ranks the correct class highly even when its final hard prediction is wrong. Its core independence assumption is violated by several strongly correlated feature pairs in this dataset (`px_height`/`px_width`, `three_g`/`four_g`), which distorts its probability calibration and hard decisions, while the overall ranking signal survives. |
| Random Forest (Ensemble) | Second-best overall (Accuracy 0.880, AUC 0.9767) and a large improvement over the single Decision Tree (0.830 → 0.880 accuracy, 0.887 → 0.977 AUC). Averaging 100 trees reduces the variance and overfitting that hurt the single tree, and produces much better-calibrated class probabilities — the clearest direct demonstration of what ensembling adds over its base learner. |
| **Overall Winner** | **Logistic Regression** — highest on all six metrics, including MCC (0.9534), which isn't inflated by class imbalance (the dataset here is perfectly balanced anyway). This is a useful counter-example to the common assumption that ensembles always win: when the true relationship between features and target is close to linear, a simple linear model can outperform more complex, higher-variance methods like Decision Trees and kNN. |

## Streamlit App Features

- CSV upload for test data
- Model selection dropdown (including a "Compare all models" view)
- Live evaluation metrics computed on the uploaded data
- Confusion matrix and full classification report per model

## How to Run Locally

```bash
pip install -r requirements.txt
python train_models.py --data train.csv
streamlit run app.py
```
