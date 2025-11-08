from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline

DATA_PATH = Path(r"C:\Users\Полина\Downloads\Human_Resources.csv")
OUT_DIR = DATA_PATH.parent / "clustering_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
drop_cols_hard = ["EmployeeNumber", "EmployeeCount", "StandardHours", "Over18"]
df = df.drop(columns=[c for c in drop_cols_hard if c in df.columns], errors="ignore")

holdout_for_interpret = [c for c in ["Attrition", "PerformanceRating"] if c in df.columns]
meta = df[holdout_for_interpret].copy() if holdout_for_interpret else None
X = df.drop(columns=holdout_for_interpret, errors="ignore") if holdout_for_interpret else df.copy()

num_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

Xa = X[num_cols].copy()

preprocB = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ],
    remainder="drop",
)

pipeB = Pipeline([("pre", preprocB)])
Xb = pipeB.fit_transform(X)

try:
    feat_names_B = pipeB.named_steps["pre"].get_feature_names_out().tolist()
except Exception:
    feat_names_B = [f"f{i}" for i in range(Xb.shape[1])]

def scan_k_and_fit(Xmat, k_range=range(2, 9), repeats=5, random_state=42):
    rows = []
    best = {"k": None, "sil": -1, "model": None, "labels": None}
    for k in k_range:
        sil_values = []
        for r in range(repeats):
            km = KMeans(n_clusters=k, n_init=10, random_state=random_state + r)
            labels = km.fit_predict(Xmat)
            sil = silhouette_score(Xmat, labels)
            sil_values.append(sil)
            if sil > best["sil"]:
                best = {"k": k, "sil": sil, "model": km, "labels": labels}
        rows.append({"k": k, "sil_mean": float(np.mean(sil_values)), "sil_std": float(np.std(sil_values))})
    return pd.DataFrame(rows), best

metrics_A, best_A = scan_k_and_fit(Xa)
metrics_B, best_B = scan_k_and_fit(Xb)

metrics_A.to_csv(OUT_DIR / "silhouette_A_raw_numeric.csv", index=False, encoding="utf-8-sig")
metrics_B.to_csv(OUT_DIR / "silhouette_B_preprocessed.csv", index=False, encoding="utf-8-sig")

def save_sil_plot(dfm, title, path):
    plt.figure(figsize=(6,4))
    plt.plot(dfm["k"], dfm["sil_mean"], marker="o")
    plt.fill_between(dfm["k"], dfm["sil_mean"]-dfm["sil_std"], dfm["sil_mean"]+dfm["sil_std"], alpha=0.2)
    plt.xlabel("k")
    plt.ylabel("Silhouette")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()

save_sil_plot(metrics_A, "Силуэт: схема A", OUT_DIR / "silhouette_A.png")
save_sil_plot(metrics_B, "Силуэт: схема B", OUT_DIR / "silhouette_B.png")

def pca_scatter(Xmat, labels, title, path):
    pca = PCA(n_components=2, random_state=42)
    XY = pca.fit_transform(Xmat)
    plt.figure(figsize=(6,5))
    plt.scatter(XY[:,0], XY[:,1], c=labels, s=10)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()

kA, labA = best_A["k"], best_A["labels"]
pca_scatter(Xa, labA, f"PCA: схема A (k={kA})", OUT_DIR / f"pca_A_k{kA}.png")

kB, labB = best_B["k"], best_B["labels"]
pca_scatter(Xb, labB, f"PCA: схема B (k={kB})", OUT_DIR / f"pca_B_k{kB}.png")

def cluster_profiles_numeric(Xdf_num, labels):
    prof = Xdf_num.copy()
    prof["cluster"] = labels
    return prof.groupby("cluster").agg("mean").reset_index()

def cluster_profiles_categorical(Xdf_cat, labels):
    if Xdf_cat is None or Xdf_cat.empty:
        return None
    tmp = Xdf_cat.copy()
    tmp["cluster"] = labels
    parts = []
    for col in Xdf_cat.columns:
        tab = tmp.groupby("cluster")[col].value_counts(normalize=True).rename("share").reset_index()
        parts.append(tab)
    return pd.concat(parts, axis=0, ignore_index=True)

profA_num = cluster_profiles_numeric(Xa, labA)
profA_num.to_csv(OUT_DIR / f"profiles_A_k{kA}_numeric_means.csv", index=False, encoding="utf-8-sig")

X_num_df = X[num_cols].copy()
X_cat_df = X[cat_cols].copy() if cat_cols else None

profB_num = cluster_profiles_numeric(X_num_df, labB)
profB_num.to_csv(OUT_DIR / f"profiles_B_k{kB}_numeric_means.csv", index=False, encoding="utf-8-sig")

profB_cat = cluster_profiles_categorical(X_cat_df, labB) if X_cat_df is not None else None
if profB_cat is not None:
    profB_cat.to_csv(OUT_DIR / f"profiles_B_k{kB}_categorical_shares.csv", index=False, encoding="utf-8-sig")

def crosstab_with_meta(labels, meta_df, col, path):
    tab = pd.crosstab(labels, meta_df[col], normalize="index")
    tab.to_csv(path, encoding="utf-8-sig")
    return tab

if meta is not None:
    for col in meta.columns:
        crosstab_with_meta(labA, meta, col, OUT_DIR / f"crosstab_A_{col}.csv")
        crosstab_with_meta(labB, meta, col, OUT_DIR / f"crosstab_B_{col}.csv")

summary = pd.DataFrame([
    {"scheme": "A_raw_numeric", "best_k": kA, "best_silhouette": float(best_A["sil"])},
    {"scheme": "B_scaled_OHE",  "best_k": kB, "best_silhouette": float(best_B["sil"])},
])
summary.to_csv(OUT_DIR / "summary_best_results.csv", index=False, encoding="utf-8-sig")

print("Готово.")
