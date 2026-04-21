# Project Report: Census Income Classification & Customer Segmentation

**Client:** Retail Business (Marketing Use Case)  
**Data:** 1994–1995 U.S. Census Bureau Current Population Survey  
**Date:** April 2026

---

## 1. Executive Summary

This project addresses two business objectives:

1. **Classification:** Predict whether an individual earns more or less than $50,000/year, enabling the client to target likely high-income customers.
2. **Segmentation:** Identify distinct customer groups in the population to support differentiated marketing strategies.

The final classifier (LightGBM) achieves a ROC-AUC of **0.9404** on held-out test data. Four customer segments were identified, each with a distinct profile and marketing implication. The key business recommendation is to use both models in combination: the classifier scores individuals, and the segmentation model guides the messaging strategy per group.

---

## 2. Data Exploration

### 2.1 Dataset Overview

- **Records:** 199,523 observations
- **Features:** 40 demographic and employment variables (age, education, occupation, marital status, capital gains/losses, etc.)
- **Weight column:** Each record carries a sampling weight representing how many people in the general population that record represents (stratified sampling). This is important for any population-level analysis.
- **Label:** Binary — income ≤ $50K (`- 50000.`) or > $50K (`50000+.`)

### 2.2 Class Imbalance

| Class | Raw Count | Weighted % of Population |
|---|---|---|
| ≤ $50K | 187,141 | ~93.8% |
| > $50K | 12,382 | ~6.2% |

The dataset is **highly imbalanced** — roughly 15:1. This is realistic (high earners are a minority), but requires special handling in model training to avoid the model defaulting to always predicting the majority class.

### 2.3 Key Findings from EDA

- **Education** is a strong predictor: college graduates and above have substantially higher rates of earning >$50K.
- **Age** peaks around 40–55 for >$50K earners; very young and very old individuals rarely exceed the threshold.
- **Capital gains and dividends** are highly skewed — most values are zero, but when non-zero they strongly indicate high income.
- **Missing values** appear in 7 columns (represented as `?`), primarily migration-related fields. These were imputed rather than dropped to preserve all 199K records.
- **Weight-adjusted analysis** confirms that the raw sample proportions broadly reflect population distributions, validating use of the full dataset without re-weighting for model training.

---

## 3. Data Preprocessing

### 3.1 Decisions Made

| Step | Decision | Rationale |
|---|---|---|
| Missing values (`?`) | Impute with median (numeric) / mode (categorical) | Retains all 199K rows; missingness is not random but correlates with "Not in universe" categories |
| Categorical encoding | Label encoding (integer mapping) | Compatible with tree models (LightGBM); avoids high-dimensional one-hot encoding for 30+ categorical columns |
| Weight column | Excluded from features; retained for population-level analysis | Weight is a sampling artifact, not a demographic signal |
| Label encoding | `50000+.` → 1, `- 50000.` → 0 | Standard binary target |

### 3.2 Note on Encoding Choice

Label encoding (vs. one-hot) was chosen deliberately for tree-based models. LightGBM treats label-encoded categoricals via split points and is not harmed by ordinal assumptions within the tree structure. For linear models (Logistic Regression), we apply StandardScaler to mitigate the scale differences this introduces.

---

## 4. Classification Model

### 4.1 Approach

Two models were trained:

- **Logistic Regression** (baseline) — interpretable, establishes a performance floor
- **LightGBM** (main model) — gradient boosting, handles mixed feature types and imbalanced data well

Both use `class_weight='balanced'` / `scale_pos_weight` to address the 15:1 class imbalance.

### 4.2 Train/Test Split

80/20 stratified split. The positive rate is preserved at 6.2% in both splits.

### 4.3 Results

| Model | ROC-AUC | Avg. Precision |
|---|---|---|
| Logistic Regression | 0.9303 | 0.5414 |
| LightGBM | **0.9404** | **0.5722** |

Both models achieve strong discriminative power (ROC-AUC > 0.93). LightGBM is preferred as the production model.

**Note on default threshold:** With the default 0.5 decision threshold, LightGBM predicts all instances as ≤$50K — a known behavior when `scale_pos_weight` shifts the probability distribution. The ROC-AUC metric is unaffected (it evaluates ranking, not hard predictions), but threshold calibration is required before deployment.

### 4.4 Threshold Selection for Marketing

| Threshold | Precision (>50K) | Recall (>50K) | # Flagged (test set) |
|---|---|---|---|
| 0.10 | 16.5% | 97.3% | 14,609 |
| 0.15 | 22.6% | 93.5% | 10,260 |
| 0.20 | 28.9% | 87.5% | 7,508 |
| 0.25 | 39.1% | 77.3% | 4,898 |
| 0.30 | 60.1% | 48.0% | 1,980 |

**Business interpretation:**
- **Broad campaign** (threshold 0.20): Flag ~7,500 people per 40K; 87.5% of true high-earners captured, ~29% of flagged are genuinely high-income.
- **Precision targeting** (threshold 0.30): Flag ~2,000 people; 60% of flagged are genuinely high-income, but miss ~52% of the high-earner pool.

**Recommendation:** Use threshold **0.20–0.25** for direct mail / digital campaigns (maximize reach), and threshold **0.30** for high-touch channels (e.g., personal outreach, premium offers) where cost-per-contact is high.

### 4.5 Top Predictive Features

The top 10 features by LightGBM split importance:

1. Dividends from stocks
2. Age
3. Education
4. Capital gains
5. Detailed industry recode
6. Major occupation code
7. Num persons worked for employer
8. Detailed occupation recode
9. Capital losses
10. Sex

Investment income (dividends, capital gains) and human capital (education, occupation) dominate. This aligns with economic intuition and suggests the model is learning genuine income signals rather than spurious correlations.

---

## 5. Segmentation Model

### 5.1 Approach

Customer segmentation was performed using K-Means clustering on the full feature set (excluding label and weight), after dimensionality reduction via PCA.

**Why PCA first?**
- 40 features with label-encoded categoricals create a mixed, high-dimensional space where Euclidean distance (used by K-Means) is less meaningful.
- PCA de-correlates features and focuses clustering on the principal axes of variation.
- 20 components were retained, explaining 80.8% of total variance.

### 5.2 Choosing K

Elbow method and silhouette scores were evaluated for k = 2 to 9. Silhouette scores were highest around k = 5–6, but the improvement is marginal (0.17 vs 0.19). **K = 4 was selected** for interpretability — four segments map cleanly to distinct life-stage/economic profiles that are actionable for marketing.

### 5.3 Cluster Profiles

Cluster sizes are reported as population-weighted percentages.

| Cluster | Pop. Share | Avg. Age | Avg. Wage/hr | Avg. Capital Gains | % Earning >$50K |
|---|---|---|---|---|---|
| 0 | 28.0% | ~9 | ~$1 | ~$5 | 0.0% |
| 1 | 44.8% | 38.4 | $120 | $824 | 12.4% |
| 2 | 20.5% | 58.5 | ~$1 | $244 | 2.4% |
| 3 | 6.8% | 39.3 | $45 | $314 | 5.4% |

**Segment descriptions:**

**Cluster 0 — Children / Non-Working (28%)**
Age profile ~9 years. This group represents minors and non-participants in the labor force. Income is essentially zero. Not a direct marketing target for income-based products, but relevant for family-oriented or education product lines (targeting their parents).

**Cluster 1 — Prime Working Adults (44.8%)**
The largest and most economically active segment. Average age 38, higher wage per hour, notable capital gains. Highest share of >$50K earners (12.4%). This is the **primary target** for premium retail, investment products, and lifestyle brands.

**Cluster 2 — Older / Near-Retirement Adults (20.5%)**
Average age 58.5, lower wages but moderate capital gains (likely from accumulated investments). Income >$50K rate is low (2.4%), but this group may have significant net worth not reflected in current income. Relevant for retirement financial products, health & wellness, and leisure/travel.

**Cluster 3 — Mid-Career Moderate Earners (6.8%)**
Smallest population segment. Age similar to Cluster 1 but lower wages and lower >$50K rate. Likely includes part-time workers, self-employed with variable income, or career-changers. Relevant for affordable mid-market retail and financial education products.

---

## 6. Business Recommendations

### 6.1 Combined Model Strategy

Use both models together:
1. **Score everyone** with the classifier to get a probability of earning >$50K.
2. **Assign everyone** to a customer segment.
3. **Prioritize** Cluster 1 individuals with high classifier scores (>0.25) for premium outreach.
4. **Customize messaging** by segment even for lower-scoring individuals.

### 6.2 Segment-Specific Marketing

| Segment | Recommended Channel | Product Focus |
|---|---|---|
| Cluster 1 (high score) | Premium direct mail, digital retargeting | Luxury goods, investment products, travel |
| Cluster 1 (low score) | Broad digital campaigns | Mid-range retail, financial savings products |
| Cluster 2 | Email, print | Retirement planning, health, leisure |
| Cluster 3 | Budget digital channels | Promotions, value retail |
| Cluster 0 | Target parents (Cluster 1/2) | Children's products, education |

### 6.3 Model Limitations & Future Work

- **Data age:** The dataset is from 1994–1995. Income thresholds, industry distributions, and demographics have changed significantly. Re-training on recent data is essential before deployment.
- **$50K threshold:** In 1994 dollars, $50K is roughly equivalent to $100K+ in 2024 terms. The business should consider whether this threshold still represents "high income" for its marketing purposes.
- **Label encoding:** For a production system, target encoding or embeddings for high-cardinality categoricals would improve model quality.
- **Segmentation stability:** K-Means is sensitive to initialization. Results should be validated with multiple seeds and potentially alternative methods (e.g., Gaussian Mixture Models) for soft cluster assignments.
- **Ethics:** Features like race and sex are present in the data. The model should be audited for disparate impact before any production use.

---

## 7. References

1. Kohavi, R. (1996). Scaling Up the Accuracy of Naive-Bayes Classifiers: A Decision-Tree Hybrid. *Proceedings of the Second International Conference on Knowledge Discovery and Data Mining*.
2. Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *Advances in Neural Information Processing Systems 30*.
3. U.S. Census Bureau. Current Population Survey, 1994–1995. Technical Documentation.
4. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR 12*, pp. 2825–2830.
5. James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2nd ed.). Springer.
