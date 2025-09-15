[DA5401] DA Lab - Assignment-4-DA25M014
Name - Jigarahemad K Shaikh, Roll Number: DA25M014

File needed for Evaluation - "DA5401_DA25M014_Assignment 4.ipynb"


#Packages and libraries used: 
numpy (numpy as np),
pandas (pandas as pd),
matplotlib,
matplotlib.pyplot as plt,
matplotlib.patches.Patch (for legend),
scikit-learn (sklearn),
model_selection: train_test_split,
linear_model: LogisticRegression,
preprocessing: StandardScaler,
pipeline: make_pipeline,
metrics: accuracy_score, precision_recall_fscore_support, confusion_matrix,roc_curve, auc, precision_recall_curve, average_precision_score




#High Level Summary, Recommendations and Answers to the Queries-


| Model          | Accuracy | Precision | Recall | F1-score |

| Baseline       | 0.9991 | 0.8267 | 0.6327 | 0.7168 |
| GMM-Only       | 0.9811 | 0.0763 | 0.8980 | 0.1406 |
| CBU+GMM        | 0.9712 | 0.0523 | 0.9184 | 0.0990 |


- Recall (minority detection) skyrocketed** from 0.63 → ~0.90 with GMM-Only and ~0.92 with CBU+GMM.  
  → The classifier sees many more minority patterns during training and becomes far less conservative.
- Precision collapsed (0.83 → 0.08 / 0.05), indicating **many more false positives** at the default decision threshold.  
  → Heavy oversampling + undersampling shifts the learned boundary; without threshold tuning, the model over-flags fraud.
- F1 fell vs. baseline because the precision drop outweighed the recall gain.

Did GMM based oversampling improve the ability to detect the minority class?
Yes. Both GMM-Only and CBU+GMM **greatly improved recall, i.e., the model detects far more frauds.  
However, the current operating point yields very low precision, which can be impractical without further tuning.

### Recommendations (actionable)
1. **Tune the decision threshold** on a validation set to hit a business-acceptable **precision floor** (e.g., ≥0.20) while keeping recall high.  
   - Plot **PR curves**; choose the point that maximizes your utility (cost of false negatives vs false positives).
2. **Partially rebalance** instead of fully: target a ratio like **1:2 or 1:3 (minority:majority)** rather than strict 1:1 to curb false positives.
3. **Calibrate probabilities** (Platt/Isotonic) before thresholding; calibration often improves precision at a given recall.
4. **Model & loss choices:** try **XGBoost/LightGBM** with `scale_pos_weight` or **logistic with class_weight='balanced'`**, or **focal loss** (if available).
5. **Quality control on synthetic data:**  
   - Limit GMM components to avoid over-fitting tiny modes;  
   - Reject low-likelihood synthetic samples;  
   - Consider **CBU with gentler undersampling** (keep more real majority data).
6. **Report cost-aware metrics** (expected cost, recall@precision≥X) and **PR-AUC**, not accuracy.

### Final recommendation
- If the **priority is minimizing missed fraud** and manual review capacity exists, proceed with **CBU+GMM**, but **raise the decision threshold** (post-calibration) to recover precision to an acceptable level.  
- If **alerts must remain precise**, use **GMM-Only or class-weighted models** with **partial rebalance + threshold tuning**; this typically yields **higher F1** than our current fully balanced setups.



Date: 15/Sept/2025
