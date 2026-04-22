import nbformat as nbf

def cell(source, cell_type='code'):
    if cell_type == 'markdown':
        return nbf.v4.new_markdown_cell(source)
    return nbf.v4.new_code_cell(source)

# ─────────────────────────────────────────────
# 01_eda.ipynb
# ─────────────────────────────────────────────
nb1 = nbf.v4.new_notebook()
nb1.cells = [

cell("# 01 — Exploratory Data Analysis & Preprocessing", 'markdown'),

cell("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (10, 5)
sns.set_theme(style='whitegrid')
"""),

cell("## 1. Load Data", 'markdown'),

cell("""
# Read column names (one per line, no index prefix)
with open('data/census-bureau.columns') as f:
    cols = [line.strip() for line in f if line.strip()]

df = pd.read_csv('data/census-bureau.data', header=None, names=cols)
print(f"Shape: {df.shape}")
df.head(3)
"""),

cell("## 2. Basic Info", 'markdown'),

cell("""
print(df.dtypes)
print("\\nNumerical columns:")
num_cols = df.select_dtypes(include='number').columns.tolist()
print(num_cols)
"""),

cell("""
# Replace '?' with NaN
df.replace(' ?', np.nan, inplace=True)
df.replace('?', np.nan, inplace=True)

missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print("Missing values per column:")
print(missing)
print(f"\\nTotal columns with missing: {len(missing)}")
"""),

cell("## 3. Label Distribution", 'markdown'),

cell("""
label_col = 'label'
df[label_col] = df[label_col].str.strip()

label_counts = df[label_col].value_counts()
print("Raw counts:")
print(label_counts)

# Weighted distribution
weight_col = 'weight'
weighted = df.groupby(label_col)[weight_col].sum()
print("\\nWeighted counts (representative of population):")
print(weighted)
print(f"\\nWeighted % earning >50K: {weighted['50000+.'] / weighted.sum() * 100:.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
label_counts.plot(kind='bar', ax=axes[0], color=['steelblue','tomato'])
axes[0].set_title('Raw Sample Count')
axes[0].set_xlabel('')
weighted.plot(kind='bar', ax=axes[1], color=['steelblue','tomato'])
axes[1].set_title('Population-Weighted Count')
axes[1].set_xlabel('')
plt.tight_layout()
plt.savefig('figures/fig_label_dist.png', dpi=100)
plt.show()
print("Class imbalance ratio:", round(label_counts.iloc[0]/label_counts.iloc[1], 1), ":1")
"""),

cell("## 4. Numerical Features", 'markdown'),

cell("""
num_features = [c for c in num_cols if c not in ['weight']]
print("Numerical features:", num_features)
df[num_features].describe().round(2)
"""),

cell("""
fig, axes = plt.subplots(2, 5, figsize=(18, 7))
axes = axes.flatten()
for i, col in enumerate(num_features[:10]):
    data = df[col].dropna()
    axes[i].hist(data, bins=40, color='steelblue', edgecolor='none', alpha=0.8)
    axes[i].set_title(col, fontsize=9)
    axes[i].set_ylabel('Count')
plt.suptitle('Numerical Feature Distributions', y=1.01)
plt.tight_layout()
plt.savefig('figures/fig_num_dist.png', dpi=100)
plt.show()
"""),

cell("## 5. Key Categorical Features", 'markdown'),

cell("""
key_cats = ['education', 'marital stat', 'sex', 'race', 'major occupation code',
            'full or part time employment stat', 'tax filer stat', 'citizenship']

fig, axes = plt.subplots(4, 2, figsize=(16, 18))
axes = axes.flatten()
for i, col in enumerate(key_cats):
    counts = df[col].value_counts().head(10)
    counts.plot(kind='barh', ax=axes[i], color='steelblue')
    axes[i].set_title(col, fontsize=9)
    axes[i].invert_yaxis()
plt.tight_layout()
plt.savefig('figures/fig_cat_dist.png', dpi=100)
plt.show()
"""),

cell("## 6. Feature vs Label", 'markdown'),

cell("""
# Income rate by education
edu_income = df.groupby('education')[label_col].apply(
    lambda x: (x == '50000+.').mean() * 100
).sort_values(ascending=False)
edu_income.plot(kind='barh', color='steelblue', figsize=(10, 6))
plt.title('% Earning >50K by Education Level')
plt.xlabel('% >50K')
plt.tight_layout()
plt.savefig('figures/fig_edu_income.png', dpi=100)
plt.show()
"""),

cell("""
# Age distribution by label
df_plot = df[['age', label_col]].dropna()
df_plot.groupby(label_col)['age'].plot(kind='hist', bins=40, alpha=0.6, legend=True)
plt.title('Age Distribution by Income Group')
plt.xlabel('Age')
plt.tight_layout()
plt.savefig('figures/fig_age_income.png', dpi=100)
plt.show()
"""),

cell("## 7. Correlation (Numerical)", 'markdown'),

cell("""
corr = df[num_features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True)
plt.title('Correlation Matrix — Numerical Features')
plt.tight_layout()
plt.savefig('figures/fig_corr.png', dpi=100)
plt.show()
"""),

cell("## 8. Preprocessing & Save", 'markdown'),

cell("""
from sklearn.preprocessing import LabelEncoder

df_clean = df.copy()

# Binary label
df_clean['label_binary'] = (df_clean[label_col] == '50000+.').astype(int)
df_clean.drop(columns=[label_col], inplace=True)

# Identify column types
cat_cols = df_clean.select_dtypes(include='object').columns.tolist()
num_feat_cols = [c for c in df_clean.select_dtypes(include='number').columns
                 if c not in ['weight', 'label_binary']]

print(f"Categorical columns: {len(cat_cols)}")
print(f"Numerical feature columns: {len(num_feat_cols)}")

# Fill missing
for c in num_feat_cols:
    df_clean[c].fillna(df_clean[c].median(), inplace=True)
for c in cat_cols:
    df_clean[c].fillna(df_clean[c].mode()[0], inplace=True)

# Label encode categoricals
le = LabelEncoder()
for c in cat_cols:
    df_clean[c] = le.fit_transform(df_clean[c].astype(str))

print("\\nFinal shape:", df_clean.shape)
print("Missing values remaining:", df_clean.isnull().sum().sum())
df_clean.head(3)
"""),

cell("""
df_clean.to_csv('data/census_preprocessed.csv', index=False)
print("Saved: census_preprocessed.csv")
print("Columns:", list(df_clean.columns))
"""),

]

# ─────────────────────────────────────────────
# 02_classification.ipynb
# ─────────────────────────────────────────────
nb2 = nbf.v4.new_notebook()
nb2.cells = [

cell("# 02 — Classification Model (Income >50K vs ≤50K)", 'markdown'),

cell("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, precision_recall_curve,
                              average_precision_score)
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

plt.rcParams['figure.figsize'] = (10, 5)
sns.set_theme(style='whitegrid')
"""),

cell("## 1. Load Data", 'markdown'),

cell("""
df = pd.read_csv('data/census_preprocessed.csv')
print("Shape:", df.shape)

WEIGHT_COL = 'weight'
TARGET = 'label_binary'
feature_cols = [c for c in df.columns if c not in [TARGET, WEIGHT_COL]]

X = df[feature_cols]
y = df[TARGET]
w = df[WEIGHT_COL]

print(f"Features: {X.shape[1]}")
print(f"Class distribution:\\n{y.value_counts()}")
print(f"\\nPositive rate: {y.mean()*100:.1f}%")
"""),

cell("## 2. Train/Test Split", 'markdown'),

cell("""
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, w, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")
print(f"Train positive rate: {y_train.mean()*100:.1f}%")
print(f"Test  positive rate: {y_test.mean()*100:.1f}%")
"""),

cell("## 3. Baseline — Logistic Regression", 'markdown'),

cell("""
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

lr = LogisticRegression(class_weight='balanced', max_iter=500, random_state=42)
lr.fit(X_train_sc, y_train)

y_pred_lr = lr.predict(X_test_sc)
y_prob_lr = lr.predict_proba(X_test_sc)[:, 1]

print("=== Logistic Regression ===")
print(classification_report(y_test, y_pred_lr, target_names=['<=50K', '>50K']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_lr):.4f}")
print(f"Avg Precision: {average_precision_score(y_test, y_prob_lr):.4f}")
"""),

cell("## 4. Main Model — LightGBM", 'markdown'),

cell("""
# Scale pos weight to handle imbalance
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight: {scale_pos:.1f}")

# Split a validation set from train (to avoid using test set for early stopping)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
)

lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=63,
    max_depth=-1,
    scale_pos_weight=scale_pos,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

lgb_model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)]
)

y_pred_lgb = lgb_model.predict(X_test)
y_prob_lgb = lgb_model.predict_proba(X_test)[:, 1]

print("\\n=== LightGBM ===")
print(classification_report(y_test, y_pred_lgb, target_names=['<=50K', '>50K']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_lgb):.4f}")
print(f"Avg Precision: {average_precision_score(y_test, y_prob_lgb):.4f}")
"""),

cell("## 5. Evaluation", 'markdown'),

cell("""
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Confusion Matrix — use calibrated threshold (0.25) not default 0.5
# Default threshold predicts all-negative due to scale_pos_weight distribution shift
THRESHOLD = 0.25
y_pred_lgb_cal = (y_prob_lgb >= THRESHOLD).astype(int)
cm = confusion_matrix(y_test, y_pred_lgb_cal)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
axes[0].set_title(f'Confusion Matrix — LightGBM (threshold={THRESHOLD})')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# ROC Curve
for name, prob in [('Logistic Reg', y_prob_lr), ('LightGBM', y_prob_lgb)]:
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)
    axes[1].plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
axes[1].plot([0,1],[0,1],'k--')
axes[1].set_xlabel('FPR'); axes[1].set_ylabel('TPR')
axes[1].set_title('ROC Curve'); axes[1].legend()

# Precision-Recall Curve
for name, prob in [('Logistic Reg', y_prob_lr), ('LightGBM', y_prob_lgb)]:
    prec, rec, _ = precision_recall_curve(y_test, prob)
    ap = average_precision_score(y_test, prob)
    axes[2].plot(rec, prec, label=f'{name} (AP={ap:.3f})')
axes[2].set_xlabel('Recall'); axes[2].set_ylabel('Precision')
axes[2].set_title('Precision-Recall Curve'); axes[2].legend()

plt.tight_layout()
plt.savefig('figures/fig_classification_eval.png', dpi=100)
plt.show()
"""),

cell("## 6. Feature Importance", 'markdown'),

cell("""
importance = pd.Series(lgb_model.feature_importances_, index=feature_cols)
top20 = importance.nlargest(20)

top20.sort_values().plot(kind='barh', color='steelblue', figsize=(10, 7))
plt.title('Top 20 Feature Importances — LightGBM')
plt.xlabel('Importance (split count)')
plt.tight_layout()
plt.savefig('figures/fig_feature_importance.png', dpi=100)
plt.show()

print("Top 10 features:")
print(importance.nlargest(10))
"""),

cell("## 7. Model Comparison Summary", 'markdown'),

cell("""
results = pd.DataFrame({
    'Model': ['Logistic Regression', 'LightGBM'],
    'ROC-AUC': [
        round(roc_auc_score(y_test, y_prob_lr), 4),
        round(roc_auc_score(y_test, y_prob_lgb), 4)
    ],
    'Avg Precision': [
        round(average_precision_score(y_test, y_prob_lr), 4),
        round(average_precision_score(y_test, y_prob_lgb), 4)
    ]
})
print(results.to_string(index=False))
"""),

cell("""
# Business threshold analysis
# For marketing: we want high recall on >50K to not miss potential customers
thresholds = np.arange(0.1, 0.9, 0.05)
records = []
for t in thresholds:
    pred = (y_prob_lgb >= t).astype(int)
    tp = ((pred==1) & (y_test==1)).sum()
    fp = ((pred==1) & (y_test==0)).sum()
    fn = ((pred==0) & (y_test==1)).sum()
    prec = tp/(tp+fp) if (tp+fp)>0 else 0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0
    records.append({'threshold': round(t,2), 'precision': round(prec,3),
                    'recall': round(rec,3), 'flagged': int(pred.sum())})

thresh_df = pd.DataFrame(records)
print("Threshold Analysis (predicting >50K):")
print(thresh_df.to_string(index=False))
"""),

cell("""
# Business recommendation
print(\"\"\"
=== Business Recommendation ===

1. Use LightGBM as the production classifier (ROC-AUC ~0.93 vs ~0.90 for LR).

2. Threshold selection depends on marketing strategy:
   - Low threshold (0.2-0.3): High recall — cast wide net, miss fewer high-income
     customers, but more low-income people included (lower precision).
   - High threshold (0.5-0.6): High precision — target only very likely high-income
     customers, smaller but higher-quality list.
   Recommended: threshold 0.20-0.25 for broad marketing campaigns (recall-focused),
   or 0.30 for high-touch precision channels. Avoid 0.35+ as the model outputs
   no positive predictions beyond that point due to score distribution.

3. Top predictive signals: capital gains, dividends, education, age, weeks worked.
   These align with intuition — investment income and education strongly predict
   high earners. Client can use these for quick manual screening as well.

4. The model was trained on 1994-1995 data — re-training on recent data is advised
   before production deployment.
\"\"\")
"""),

]

# ─────────────────────────────────────────────
# 03_segmentation.ipynb
# ─────────────────────────────────────────────
nb3 = nbf.v4.new_notebook()
nb3.cells = [

cell("# 03 — Customer Segmentation Model", 'markdown'),

cell("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score

plt.rcParams['figure.figsize'] = (10, 5)
sns.set_theme(style='whitegrid')
"""),

cell("## 1. Load Data", 'markdown'),

cell("""
df = pd.read_csv('data/census_preprocessed.csv')
print("Shape:", df.shape)

WEIGHT_COL = 'weight'
TARGET = 'label_binary'

# Features only (drop label and weight for unsupervised)
feature_cols = [c for c in df.columns if c not in [TARGET, WEIGHT_COL]]
X = df[feature_cols].copy()
w = df[WEIGHT_COL].copy()
y = df[TARGET].copy()

print(f"Features used for segmentation: {X.shape[1]}")
"""),

cell("## 2. Scale & PCA", 'markdown'),

cell("""
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA for visualization and noise reduction
pca_full = PCA(random_state=42)
pca_full.fit(X_scaled)

# Explained variance
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n_comp_90 = np.argmax(cumvar >= 0.90) + 1
print(f"Components to explain 90% variance: {n_comp_90}")

plt.figure(figsize=(10, 4))
plt.plot(cumvar, marker='o', markersize=3)
plt.axhline(0.90, color='red', linestyle='--', label='90% threshold')
plt.axvline(n_comp_90, color='orange', linestyle='--', label=f'{n_comp_90} components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA — Cumulative Explained Variance')
plt.legend()
plt.tight_layout()
plt.savefig('figures/fig_pca_variance.png', dpi=100)
plt.show()
"""),

cell("""
# Use top components for clustering (balance info vs noise)
N_COMPONENTS = min(n_comp_90, 20)
pca = PCA(n_components=N_COMPONENTS, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"Using {N_COMPONENTS} PCA components, explaining {pca.explained_variance_ratio_.sum()*100:.1f}% variance")
"""),

cell("## 3. Choose K — Elbow + Silhouette", 'markdown'),

cell("""
K_range = range(2, 10)
inertias = []
sil_scores = []

# Use a sample for silhouette (expensive on full 200k)
sample_idx = np.random.choice(len(X_pca), size=10000, replace=False)
X_sample = X_pca[sample_idx]

for k in K_range:
    km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=5, batch_size=5000)
    km.fit(X_pca)
    inertias.append(km.inertia_)
    labels_sample = km.predict(X_sample)
    sil = silhouette_score(X_sample, labels_sample, sample_size=5000, random_state=42)
    sil_scores.append(sil)
    print(f"k={k}: inertia={km.inertia_:.0f}, silhouette={sil:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(list(K_range), inertias, 'o-', color='steelblue')
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method')

axes[1].plot(list(K_range), sil_scores, 'o-', color='tomato')
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score')

plt.tight_layout()
plt.savefig('figures/fig_kmeans_selection.png', dpi=100)
plt.show()
"""),

cell("## 4. Final K-Means Model", 'markdown'),

cell("""
# Choose k based on elbow + silhouette (typically 4-5 for this dataset)
BEST_K = 4

km_final = MiniBatchKMeans(n_clusters=BEST_K, random_state=42, n_init=10, batch_size=5000)
df['cluster'] = km_final.fit_predict(X_pca)

print("Cluster sizes:")
print(df['cluster'].value_counts().sort_index())
print("\\nWeighted cluster sizes (population-representative):")
weighted_sizes = df.groupby('cluster')[WEIGHT_COL].sum()
print((weighted_sizes / weighted_sizes.sum() * 100).round(1).astype(str) + '%')
"""),

cell("## 5. PCA 2D Visualization", 'markdown'),

cell("""
# Project to 2D for visualization
pca2d = PCA(n_components=2, random_state=42)
X_2d = pca2d.fit_transform(X_scaled)

sample_n = 5000
idx = np.random.choice(len(X_2d), sample_n, replace=False)
colors = ['#2196F3','#FF5722','#4CAF50','#9C27B0','#FF9800','#00BCD4']

plt.figure(figsize=(10, 7))
for c in range(BEST_K):
    mask = df['cluster'].values[idx] == c
    plt.scatter(X_2d[idx][mask, 0], X_2d[idx][mask, 1],
                c=colors[c], label=f'Cluster {c}', alpha=0.5, s=10)
plt.xlabel(f'PC1 ({pca2d.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca2d.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('Customer Segments — PCA 2D Projection')
plt.legend(markerscale=3)
plt.tight_layout()
plt.savefig('figures/fig_clusters_2d.png', dpi=100)
plt.show()
"""),

cell("## 6. Cluster Profiling", 'markdown'),

cell("""
# Re-attach original (encoded) features for profiling
profile_cols_num = ['age', 'education', 'wage per hour', 'capital gains',
                    'capital losses', 'dividends from stocks',
                    'weeks worked in year', 'num persons worked for employer']
profile_cols_num = [c for c in profile_cols_num if c in df.columns]

# Weighted mean per cluster for numerical features
def weighted_mean(group, weight_col):
    return (group[profile_cols_num].multiply(group[weight_col], axis=0)).sum() / group[weight_col].sum()

cluster_profiles = df.groupby('cluster').apply(lambda g: weighted_mean(g, WEIGHT_COL))
print("Weighted cluster means (numerical features):")
print(cluster_profiles.round(2))
"""),

cell("""
# Income rate per cluster (weighted)
income_rate = df.groupby('cluster').apply(
    lambda g: np.average(g[TARGET], weights=g[WEIGHT_COL]) * 100
)
print("\\nWeighted % earning >50K per cluster:")
print(income_rate.round(1))

plt.figure(figsize=(8, 4))
income_rate.plot(kind='bar', color=colors[:BEST_K], edgecolor='white')
plt.title('% Earning >50K by Cluster')
plt.xlabel('Cluster')
plt.ylabel('% >50K')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('figures/fig_cluster_income.png', dpi=100)
plt.show()
"""),

cell("""
# Heatmap of cluster profiles
fig, ax = plt.subplots(figsize=(12, 5))
profile_norm = (cluster_profiles - cluster_profiles.mean()) / (cluster_profiles.std() + 1e-9)
sns.heatmap(profile_norm.T, annot=cluster_profiles.T.round(1), fmt='g',
            cmap='RdYlGn', center=0, ax=ax, linewidths=0.5)
ax.set_title('Cluster Profiles — Z-scored Means (annotations = actual mean)')
ax.set_xlabel('Cluster')
plt.tight_layout()
plt.savefig('figures/fig_cluster_heatmap.png', dpi=100)
plt.show()
"""),

cell("## 7. Segment Descriptions & Business Recommendations", 'markdown'),

cell("""
print(\"\"\"
Based on weighted cluster profiles and income rates:

Cluster 0 — Children / Non-Working (28.0% of population)
  Avg age: ~9 years. Represents minors and non-labor-force participants.
  Wage: ~$1/hr  |  Capital gains: ~$5  |  Income >50K: 0.0%
  Marketing angle: Not a direct target; reach via parents (Cluster 1/2).
  Relevant for: children's products, education savings plans, family offers.

Cluster 1 — Prime Working Adults (44.8% of population)  [LARGEST SEGMENT]
  Avg age: 38 years. Full-time workers with meaningful investment income.
  Wage: ~$120/hr  |  Capital gains: ~$824  |  Income >50K: 12.4%
  Marketing angle: PRIMARY acquisition target for premium retail.
  Relevant for: luxury goods, investment products, travel, career development.

Cluster 2 — Older / Near-Retirement Adults (20.5% of population)
  Avg age: 58.5 years. Low wages but notable dividends ($493/yr avg).
  Wage: ~$1/hr  |  Capital gains: ~$244  |  Income >50K: 2.4%
  Note: Low current income may mask significant accumulated net worth.
  Relevant for: retirement planning, health & wellness, leisure/travel.

Cluster 3 — Mid-Career Moderate Earners (6.8% of population)
  Avg age: 39 years. Works part-year; lower wages than Cluster 1.
  Wage: ~$45/hr  |  Capital gains: ~$314  |  Income >50K: 5.4%
  Likely includes part-time workers, self-employed, or career-changers.
  Relevant for: mid-market retail, financial education, flexible benefit products.

=== Strategic Recommendations ===

1. COMBINE with classifier: Score each individual with the classifier and
   cross-reference with their cluster. Cluster 1 members with classifier
   score >0.25 are the highest-value targets for premium outreach.

2. Cluster 1 (44.8%) is the dominant addressable segment — broad campaigns
   here are cost-effective AND reach the highest concentration of >50K earners.

3. Cluster 2 has low income rate but high dividend income, suggesting hidden
   wealth. Suitable for net-worth-based (vs income-based) product targeting.

4. Cluster 0 (children) can be reached indirectly via Cluster 1/2 parents —
   family bundle messaging is the right channel.

5. Re-run segmentation annually as economic conditions shift.
\"\"\")
"""),

cell("---\\n## 8. Enhancement — Re-segmentation on Working-Age Population", 'markdown'),

cell("""
# Cluster 0 (children, avg age ~9) creates an easy split that dominates the
# silhouette score. Here we re-cluster only the working-age population (age >= 16)
# and compare KMeans against a Gaussian Mixture Model (GMM).
# Original 4-cluster results above are unchanged.
from sklearn.mixture import GaussianMixture

df_work = df[df['age'] >= 16].copy()
w_work = df_work[WEIGHT_COL]
y_work = df_work[TARGET]
X_work = df_work[feature_cols].copy()

print(f"Full dataset:          {len(df):,} records")
print(f"Working-age (age>=16): {len(df_work):,} records")
print(f"Excluded (children):   {len(df)-len(df_work):,} records ({(1-len(df_work)/len(df))*100:.1f}%)")
print(f"Working-age >50K rate: {y_work.mean()*100:.1f}%")
"""),

cell("""
scaler_w = StandardScaler()
X_work_scaled = scaler_w.fit_transform(X_work)

pca_w = PCA(n_components=20, random_state=42)
X_work_pca = pca_w.fit_transform(X_work_scaled)
print(f"PCA 20 components explain {pca_w.explained_variance_ratio_.sum()*100:.1f}% variance (working-age)")
"""),

cell("""
K_range2 = range(2, 9)
inertias2, sil_km2, sil_gmm2 = [], [], []

sample_idx2 = np.random.choice(len(X_work_pca), size=10000, replace=False)
X_sample2 = X_work_pca[sample_idx2]

for k in K_range2:
    km2 = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=5, batch_size=5000)
    km2.fit(X_work_pca)
    inertias2.append(km2.inertia_)
    lbl_km = km2.predict(X_sample2)
    sil_km2.append(silhouette_score(X_sample2, lbl_km, sample_size=5000, random_state=42))

    gmm = GaussianMixture(n_components=k, covariance_type='diag', random_state=42, max_iter=100)
    gmm.fit(X_work_pca)
    lbl_gmm = gmm.predict(X_sample2)
    sil_gmm2.append(silhouette_score(X_sample2, lbl_gmm, sample_size=5000, random_state=42))

    print(f"k={k}: KMeans sil={sil_km2[-1]:.4f}  |  GMM sil={sil_gmm2[-1]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(list(K_range2), inertias2, 'o-', color='steelblue')
axes[0].set_xlabel('k'); axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow — Working-Age KMeans')
axes[1].plot(list(K_range2), sil_km2, 'o-', color='steelblue', label='KMeans')
axes[1].plot(list(K_range2), sil_gmm2, 's--', color='tomato', label='GMM')
axes[1].set_xlabel('k'); axes[1].set_ylabel('Silhouette')
axes[1].set_title('Silhouette — KMeans vs GMM (Working-Age)')
axes[1].legend()
plt.tight_layout()
plt.savefig('figures/fig_enhance_kselection.png', dpi=100)
plt.show()
"""),

cell("""
best_k2 = list(K_range2)[int(np.argmax(sil_km2))]
best_k_gmm = list(K_range2)[int(np.argmax(sil_gmm2))]
print(f"Best KMeans k: {best_k2}  (silhouette={max(sil_km2):.4f})")
print(f"Best GMM k:    {best_k_gmm}  (silhouette={max(sil_gmm2):.4f})")

km_work = MiniBatchKMeans(n_clusters=best_k2, random_state=42, n_init=10, batch_size=5000)
df_work['cluster_km'] = km_work.fit_predict(X_work_pca)

gmm_final = GaussianMixture(n_components=best_k_gmm, covariance_type='diag', random_state=42, max_iter=200)
gmm_final.fit(X_work_pca)
df_work['cluster_gmm'] = gmm_final.predict(X_work_pca)

print("\\nKMeans cluster sizes (working-age):")
print(df_work['cluster_km'].value_counts().sort_index())
print("\\nGMM cluster sizes (working-age):")
print(df_work['cluster_gmm'].value_counts().sort_index())
"""),

cell("""
profile_cols2 = ['age', 'education', 'wage per hour', 'capital gains',
                 'dividends from stocks', 'weeks worked in year',
                 'num persons worked for employer']
profile_cols2 = [c for c in profile_cols2 if c in df_work.columns]

def weighted_mean2(df_g, wcol, pcols):
    return (df_g[pcols].multiply(df_g[wcol], axis=0)).sum() / df_g[wcol].sum()

print("=== KMeans profiles (working-age) ===")
km_prof = df_work.groupby('cluster_km').apply(lambda g: weighted_mean2(g, WEIGHT_COL, profile_cols2))
km_income = df_work.groupby('cluster_km').apply(
    lambda g: np.average(g[TARGET], weights=g[WEIGHT_COL]) * 100)
km_share = df_work.groupby('cluster_km')[WEIGHT_COL].sum()
km_share = (km_share / km_share.sum() * 100).round(1)
km_prof['% >50K'] = km_income.round(1)
km_prof['pop_share%'] = km_share
print(km_prof.round(2))

print("\\n=== GMM profiles (working-age) ===")
gmm_prof = df_work.groupby('cluster_gmm').apply(lambda g: weighted_mean2(g, WEIGHT_COL, profile_cols2))
gmm_income = df_work.groupby('cluster_gmm').apply(
    lambda g: np.average(g[TARGET], weights=g[WEIGHT_COL]) * 100)
gmm_share = df_work.groupby('cluster_gmm')[WEIGHT_COL].sum()
gmm_share = (gmm_share / gmm_share.sum() * 100).round(1)
gmm_prof['% >50K'] = gmm_income.round(1)
gmm_prof['pop_share%'] = gmm_share
print(gmm_prof.round(2))
"""),

cell("""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
km_income.plot(kind='bar', ax=axes[0], color='steelblue', edgecolor='white')
axes[0].set_title(f'KMeans (k={best_k2}) — % >50K per Cluster\\n(Working-Age Population)')
axes[0].set_xlabel('Cluster'); axes[0].set_ylabel('% Earning >50K')
axes[0].tick_params(rotation=0)

gmm_income.plot(kind='bar', ax=axes[1], color='tomato', edgecolor='white')
axes[1].set_title(f'GMM (k={best_k_gmm}) — % >50K per Cluster\\n(Working-Age Population)')
axes[1].set_xlabel('Cluster'); axes[1].set_ylabel('% Earning >50K')
axes[1].tick_params(rotation=0)

plt.tight_layout()
plt.savefig('figures/fig_enhance_income.png', dpi=100)
plt.show()
"""),

cell("""
sil_original = sil_scores[2]  # k=4, index 2 in K_range(2..9)
print("=== Silhouette Score Comparison ===")
print(f"Original  KMeans k=4  (full data, incl. children): {sil_original:.4f}")
print(f"Enhanced  KMeans k={best_k2}  (working-age only):        {max(sil_km2):.4f}")
print(f"Enhanced  GMM    k={best_k_gmm}  (working-age only):        {max(sil_gmm2):.4f}")
print(f"\\nKMeans improved: {max(sil_km2) > sil_original}")
print(f"GMM improved:    {max(sil_gmm2) > sil_original}")
"""),

]

# Write notebooks
for name, nb in [('01_eda.ipynb', nb1), ('02_classification.ipynb', nb2), ('03_segmentation.ipynb', nb3)]:
    with open(name, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Created: {name}")
