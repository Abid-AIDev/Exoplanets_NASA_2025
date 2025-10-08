# -*- coding: utf-8 -*-
"""Exoplanet Analysis Notebook - Fixed Version"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier

# ---- Settings ----
sns.set_style("whitegrid")
np.random.seed(42)

output_dir = 'output'  # Notebook-safe folder
# Clear output directory before each run
if os.path.exists(output_dir):
    for f in os.listdir(output_dir):
        fp = os.path.join(output_dir, f)
        if os.path.isfile(fp) or os.path.islink(fp):
            os.unlink(fp)
        elif os.path.isdir(fp):
            import shutil
            shutil.rmtree(fp)
os.makedirs(output_dir, exist_ok=True)

# ---- Create dataset ----
n_samples = 300
df = pd.DataFrame({
    'period_days': np.random.lognormal(2.2, 0.9, n_samples),
    'duration_hours': np.random.lognormal(1.0, 0.5, n_samples),
    'depth_ppm': np.random.lognormal(6.0, 0.9, n_samples),
    'planet_radius_rearth': np.random.lognormal(0.45, 0.4, n_samples),
    'score': np.random.uniform(0, 40, n_samples),
    'duty_cycle': np.random.uniform(0, 0.12, n_samples),
    'transit_frequency': np.random.uniform(0, 0.5, n_samples),
})

# ---- Create labels ----
labels = []
for i in range(len(df)):
    if df.loc[i, 'score'] > 26 and df.loc[i, 'period_days'] < 60:
        labels.append('CONFIRMED')
    elif df.loc[i, 'score'] < 10:
        labels.append('FALSE POSITIVE')
    else:
        labels.append('CANDIDATE')
df['label_raw'] = labels

colors_dict = {'CONFIRMED': 'green', 'FALSE POSITIVE': 'red', 'CANDIDATE': 'orange'}
df['color'] = df['label_raw'].map(colors_dict)

le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label_raw'])



# ----------------------
# FIG 1: 3D Linear Regression (Period, Depth -> Radius)
# ----------------------
X_lr = df[['period_days', 'depth_ppm']].values
y_lr = df['planet_radius_rearth'].values
lr = LinearRegression().fit(X_lr, y_lr)
r2 = lr.score(X_lr, y_lr)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111, projection='3d')

for lab in df['label_raw'].unique():
    mask = df['label_raw']==lab
    ax.scatter(df[mask]['period_days'], df[mask]['depth_ppm'], df[mask]['planet_radius_rearth'],
               label=lab, s=30, alpha=0.8)

period_range = np.linspace(df['period_days'].min(), df['period_days'].max(), 30)
depth_range = np.linspace(df['depth_ppm'].min(), df['depth_ppm'].max(), 30)
P_mesh, D_mesh = np.meshgrid(period_range, depth_range)
pred = lr.predict(np.c_[P_mesh.ravel(), D_mesh.ravel()]).reshape(P_mesh.shape)
ax.plot_surface(P_mesh, D_mesh, pred, alpha=0.3, cmap='viridis', linewidth=0, antialiased=True)

ax.set_xlabel('Period (days)')
ax.set_ylabel('Depth (ppm)')
ax.set_zlabel('Radius (R⊕)')
ax.set_title(f"Linear Regression Surface (R² = {r2:.3f})")
ax.legend()
plt.savefig(os.path.join(output_dir, 'fig1_linear_regression.png'), dpi=300, bbox_inches='tight')
plt.close()

# ----------------------
# FIG 2: Logistic Regression Decision Boundary (Score, Radius)
# ----------------------
df['is_planet'] = (df['label_raw']=='CONFIRMED').astype(int)
X_log = df[['score','planet_radius_rearth']].values
scaler = StandardScaler()
X_log_s = scaler.fit_transform(X_log)
y_log = df['is_planet'].values
log = LogisticRegression(max_iter=1000).fit(X_log_s, y_log)
acc = log.score(X_log_s, y_log)

xx = np.linspace(X_log_s[:,0].min()-0.8, X_log_s[:,0].max()+0.8, 300)
yy = np.linspace(X_log_s[:,1].min()-0.8, X_log_s[:,1].max()+0.8, 300)
XX, YY = np.meshgrid(xx, yy)
Z = log.predict_proba(np.c_[XX.ravel(), YY.ravel()])[:,1].reshape(XX.shape)

plt.figure(figsize=(8,6))
plt.contourf(XX, YY, Z, levels=25, cmap='RdYlGn', alpha=0.6)
for lab in df['label_raw'].unique():
    mask = df['label_raw']==lab
    pts = scaler.transform(df[mask][['score','planet_radius_rearth']].values)
    plt.scatter(pts[:,0], pts[:,1], label=lab, s=40, edgecolor='k')

plt.title(f'Logistic Regression (accuracy {acc:.2f})')
plt.xlabel('Score (standardized)')
plt.ylabel('Radius (standardized)')
plt.legend()
plt.savefig(os.path.join(output_dir, 'fig2_logistic_regression.png'), dpi=300, bbox_inches='tight')
plt.close()

# ----------------------
# FIG 3: PCA (3D) + KMeans clustering
# ----------------------
features = ['period_days','depth_ppm','planet_radius_rearth','score','duty_cycle','transit_frequency','duration_hours']
Xf = df[features].values
Xf_s = StandardScaler().fit_transform(Xf)
pca = PCA(n_components=3).fit(Xf_s)
X_p = pca.transform(Xf_s)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(Xf_s)
clusters = kmeans.labels_

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
for lab in df['label_raw'].unique():
    mask = df['label_raw']==lab
    ax.scatter(X_p[mask,0], X_p[mask,1], X_p[mask,2], label=lab, s=30)

ax.set_title('PCA 3D (static snapshot)')
ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3')
ax.legend()
plt.savefig(os.path.join(output_dir, 'fig3_pca_kmeans.png'), dpi=300, bbox_inches='tight')
plt.close()

# ----------------------
# FIG 4: Decision Tree
# ----------------------
features_dt = ['score','planet_radius_rearth','period_days','depth_ppm']
Xdt = df[features_dt].values
ydt = df['label_encoded'].values
dt = DecisionTreeClassifier(max_depth=4, random_state=42, min_samples_split=8).fit(Xdt, ydt)
dt_acc = dt.score(Xdt, ydt)

rules = export_text(dt, feature_names=features_dt, max_depth=4)
with open(os.path.join(output_dir, 'fig4_decision_tree_rules.txt'), 'w') as f:
    f.write(rules)

plt.figure(figsize=(12,6))
plot_tree(dt, feature_names=features_dt, class_names=le.classes_, filled=True, rounded=True, proportion=True)
plt.title(f"Decision Tree (accuracy {dt_acc:.2f})")
plt.savefig(os.path.join(output_dir, 'fig4_decision_tree.png'), dpi=300, bbox_inches='tight')
plt.close()

fi = dt.feature_importances_
plt.figure(figsize=(8,5))
plt.bar(features_dt, fi)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Decision Tree: Feature Importance')
plt.savefig(os.path.join(output_dir, 'fig4_decision_tree_importance.png'), dpi=300, bbox_inches='tight')
plt.close()

# ----------------------
# FIG 5: Random Forest Feature Importance
# ----------------------
features_rf = features
Xrf = df[features_rf].values
yrf = df['label_encoded'].values
rf = RandomForestClassifier(n_estimators=120, random_state=42).fit(Xrf, yrf)
fi_rf = rf.feature_importances_
plt.figure(figsize=(8,5))
plt.barh(features_rf[::-1], fi_rf[::-1])
plt.xlabel('Importance')
plt.title('Random Forest: Feature Importance (static)')
plt.savefig(os.path.join(output_dir, 'fig5_random_forest_importance.png'), dpi=300, bbox_inches='tight')
plt.close()

# ----------------------
# FIG 6: Correlation Heatmap
# ----------------------
corr = df[features_rf].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0)
plt.title('Correlation (static seaborn)')
plt.savefig(os.path.join(output_dir, 'fig6_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()

# ----------------------
# FIG 7: Distribution Analysis (Violin / Box)
# ----------------------
plt.figure(figsize=(12,8))
plot_cols = ['score','planet_radius_rearth','period_days','depth_ppm']
for i, c in enumerate(plot_cols, 1):
    plt.subplot(2,2,i)
    sns.violinplot(x='label_raw', y=c, data=df, palette=colors_dict, inner='quartile')
    plt.title(c)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig7_violin_plots.png'), dpi=300, bbox_inches='tight')
plt.close()
