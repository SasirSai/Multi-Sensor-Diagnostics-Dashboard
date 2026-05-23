# 🔬 Advanced Diagnostics Performance Analytics Report

This report details the exhaustive statistical classification metrics computed across all four diagnostic models on completely unseen, leak-free holdout test motor runs.

## 📊 Model Engine: Proposed Pure RF

| Fault Category | TP | TN | FP | FN | Sensitivity (Recall) % | Specificity % | Precision % | FPR % | FNR % | F1-Score % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Normal** | 0 | 9684 | 0 | 0 | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **BPFI** | 1842 | 7842 | 0 | 0 | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 | 100.0 |
| **BPFO** | 1842 | 7842 | 0 | 0 | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 | 100.0 |
| **Misalign** | 2716 | 6684 | 0 | 284 | 90.53 | 100.0 | 100.0 | 0.0 | 9.47 | 95.03 |
| **Unbalance** | 3000 | 6400 | 284 | 0 | 100.0 | 95.75 | 91.35 | 4.25 | 0.0 | 95.48 |

---

## 📊 Model Engine: Advanced Hybrid (RF-IF)

| Fault Category | TP | TN | FP | FN | Sensitivity (Recall) % | Specificity % | Precision % | FPR % | FNR % | F1-Score % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Normal** | 0 | 9684 | 0 | 0 | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **BPFI** | 1842 | 7842 | 0 | 0 | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 | 100.0 |
| **BPFO** | 1842 | 7842 | 0 | 0 | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 | 100.0 |
| **Misalign** | 2766 | 6684 | 0 | 234 | 92.2 | 100.0 | 100.0 | 0.0 | 7.8 | 95.94 |
| **Unbalance** | 3000 | 6450 | 234 | 0 | 100.0 | 96.5 | 92.76 | 3.5 | 0.0 | 96.25 |

---

## 📊 Model Engine: Conventional Hybrid (RF-IF)

| Fault Category | TP | TN | FP | FN | Sensitivity (Recall) % | Specificity % | Precision % | FPR % | FNR % | F1-Score % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Normal** | 0 | 9684 | 0 | 0 | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **BPFI** | 1842 | 7842 | 0 | 0 | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 | 100.0 |
| **BPFO** | 1842 | 7842 | 0 | 0 | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 | 100.0 |
| **Misalign** | 2582 | 6684 | 0 | 418 | 86.07 | 100.0 | 100.0 | 0.0 | 13.93 | 92.51 |
| **Unbalance** | 3000 | 6266 | 418 | 0 | 100.0 | 93.75 | 87.77 | 6.25 | 0.0 | 93.49 |

---

## 📊 Model Engine: Optimized Gradient Boosting (GB-IF)

| Fault Category | TP | TN | FP | FN | Sensitivity (Recall) % | Specificity % | Precision % | FPR % | FNR % | F1-Score % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Normal** | 0 | 9431 | 253 | 0 | 0.0 | 97.39 | 0.0 | 2.61 | 0.0 | 0.0 |
| **BPFI** | 1842 | 7842 | 0 | 0 | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 | 100.0 |
| **BPFO** | 1842 | 7842 | 0 | 0 | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 | 100.0 |
| **Misalign** | 2000 | 6684 | 0 | 1000 | 66.67 | 100.0 | 100.0 | 0.0 | 33.33 | 80.0 |
| **Unbalance** | 3000 | 5937 | 747 | 0 | 100.0 | 88.82 | 80.06 | 11.18 | 0.0 | 88.93 |

---

