"""
Genera las figuras y cifras de la presentacion "Conformal Prediction".

Tres metodos:
  1. Split conformal (banda / umbral fijo)
  2. CQR (banda adaptativa)
  3. Mondrian (un umbral por clase)

Parte A: regresion sintetica heterocedastica con XGBoost (split vs. CQR).
Parte B: fraude (dataset del curso kartik2112/fraud-detection) con XGBClassifier,
         evaluado con las metricas del curso: recall, precision, alertas y matriz de confusion.
Parte C: variabilidad de la cobertura segun el tamano de calibracion (teorica).

Salida: ../figuras/*.png y ../figuras/resultados.json
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)
OUT = Path(__file__).resolve().parents[1] / "figuras"
OUT.mkdir(exist_ok=True)
for old in ("b_fraude_cobertura_por_clase.png", "b_scores_calibracion.png", "b_cobertura_vs_alpha.png",
            "b_reglas_recall_costo.png"):
    (OUT / old).unlink(missing_ok=True)

BLUE, ORANGE, AQUA, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e34948"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "Segoe UI", "font.size": 12, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})
pct = plt.FuncFormatter(lambda v, _: f"{v:.0%}")
results: dict = {}


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """k-esimo score mas pequeno, con k = ceil((n+1)(1-alpha))."""
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.inf
    return float(np.sort(scores)[k - 1])


# =============================================================================
# PARTE A. Regresion: split conformal (banda fija) vs. CQR (banda adaptativa)
# =============================================================================
ALPHA = 0.10
n = 3000
x = rng.uniform(0, 10, n)
y = x * np.sin(x) + rng.normal(0, 0.15 + 0.35 * x)
X = x.reshape(-1, 1)
idx = rng.permutation(n)
tr, cal, te = idx[:1500], idx[1500:2250], idx[2250:]

XGB_REG = dict(n_estimators=300, learning_rate=0.05, max_depth=3, min_child_weight=20, random_state=RANDOM_STATE)
mean_model = XGBRegressor(**XGB_REG).fit(X[tr], y[tr])
q_hat = conformal_quantile(np.abs(y[cal] - mean_model.predict(X[cal])), ALPHA)
lo_m = XGBRegressor(objective="reg:quantileerror", quantile_alpha=ALPHA / 2, **XGB_REG).fit(X[tr], y[tr])
hi_m = XGBRegressor(objective="reg:quantileerror", quantile_alpha=1 - ALPHA / 2, **XGB_REG).fit(X[tr], y[tr])
q_cqr = conformal_quantile(np.maximum(lo_m.predict(X[cal]) - y[cal], y[cal] - hi_m.predict(X[cal])), ALPHA)

pred_te = mean_model.predict(X[te])
cov_split = np.mean(np.abs(y[te] - pred_te) <= q_hat)
lo_te, hi_te = lo_m.predict(X[te]) - q_cqr, hi_m.predict(X[te]) + q_cqr
cov_cqr = np.mean((y[te] >= lo_te) & (y[te] <= hi_te))
raw_cov = np.mean((y[te] >= lo_m.predict(X[te])) & (y[te] <= hi_m.predict(X[te])))
cond = pd.DataFrame({
    "bin": pd.cut(x[te], [0, 2.5, 5, 7.5, 10]),
    "split": np.abs(y[te] - pred_te) <= q_hat,
    "cqr": (y[te] >= lo_te) & (y[te] <= hi_te),
}).groupby("bin", observed=True)[["split", "cqr"]].mean()
results["regresion"] = {
    "q_hat_split": q_hat, "cobertura_split": cov_split, "ancho_split": 2 * q_hat,
    "cobertura_banda_cuantilica_sin_calibrar": raw_cov, "q_hat_cqr": q_cqr,
    "cobertura_cqr": cov_cqr, "ancho_medio_cqr": float(np.mean(hi_te - lo_te)),
    "cobertura_condicional": {str(k): v for k, v in cond.to_dict(orient="index").items()},
}

grid = np.linspace(0, 10, 400).reshape(-1, 1)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
for ax, title, lo, hi, color in [
    (axes[0], f"Split conformal: banda fija (cobertura {cov_split:.1%})",
     mean_model.predict(grid) - q_hat, mean_model.predict(grid) + q_hat, BLUE),
    (axes[1], f"CQR: banda adaptativa (cobertura {cov_cqr:.1%})",
     lo_m.predict(grid) - q_cqr, hi_m.predict(grid) + q_cqr, AQUA),
]:
    ax.scatter(x[te], y[te], s=8, color=MUTED, alpha=0.45, linewidths=0, label="datos de prueba")
    ax.fill_between(grid.ravel(), lo, hi, color=color, alpha=0.22, linewidth=0, label="intervalo 90 %")
    ax.plot(grid, lo, color=color, lw=2)
    ax.plot(grid, hi, color=color, lw=2)
    ax.set(title=title, xlabel="x")
axes[0].set_ylabel("y")
axes[0].legend(loc="upper left")
plt.tight_layout()
plt.savefig(OUT / "a_regresion_split_vs_cqr.png", dpi=200)
plt.close()

fig, ax = plt.subplots(figsize=(7, 4))
pos = np.arange(len(cond))
ax.bar(pos - 0.2, cond["split"], 0.38, color=BLUE, label="Split (banda fija)")
ax.bar(pos + 0.2, cond["cqr"], 0.38, color=AQUA, label="CQR (adaptativa)")
ax.axhline(1 - ALPHA, color=INK, ls="--", lw=1.5, label="objetivo 90 %")
ax.set_xticks(pos, [f"x en {b}" for b in cond.index.astype(str)])
ax.set(ylim=(0.5, 1.05), ylabel="cobertura")
ax.set_title("Cobertura por zona de x (condicional)", pad=34)
ax.yaxis.set_major_formatter(pct)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=11)
plt.tight_layout()
plt.savefig(OUT / "a_cobertura_condicional.png", dpi=200)
plt.close()

# =============================================================================
# PARTE B. Fraude con XGBoost: umbrales clasicos vs. split conformal vs. Mondrian
# =============================================================================
DATA = Path.home() / ".cache/kagglehub/datasets/kartik2112/fraud-detection/versions/1"
if not (DATA / "fraudTrain.csv").exists():
    import kagglehub  # dataset publico; se descarga la primera vez
    DATA = Path(kagglehub.dataset_download("kartik2112/fraud-detection"))
COLS = ["trans_date_trans_time", "category", "amt", "gender", "lat", "long", "city_pop", "dob",
        "merch_lat", "merch_long", "is_fraud"]


def features(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["trans_date_trans_time"])
    lat1, lon1, lat2, lon2 = map(np.radians, (df["lat"], df["long"], df["merch_lat"], df["merch_long"]))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return pd.DataFrame({
        "log_amt": np.log1p(df["amt"]), "hour": ts.dt.hour, "day_of_week": ts.dt.dayofweek,
        "age": (ts - pd.to_datetime(df["dob"])).dt.days / 365.25, "log_city_pop": np.log1p(df["city_pop"]),
        "distance_km": 6371 * 2 * np.arcsin(np.sqrt(a)),
        "category": df["category"].astype("category"), "gender": df["gender"].astype("category"),
    })


hist = pd.read_csv(DATA / "fraudTrain.csv", usecols=COLS)
hist, _ = train_test_split(hist, train_size=300_000, stratify=hist["is_fraud"], random_state=RANDOM_STATE)
Xh, yh = features(hist), hist["is_fraud"].to_numpy()
X_tr, X_rest, y_tr, y_rest = train_test_split(Xh, yh, test_size=0.4, stratify=yh, random_state=RANDOM_STATE)
X_cal, X_te, y_cal, y_te = train_test_split(X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=RANDOM_STATE)

SCALE_POS_WEIGHT = (y_tr == 0).sum() / y_tr.sum()
clf = XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                    scale_pos_weight=SCALE_POS_WEIGHT, eval_metric="aucpr", enable_categorical=True,
                    tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE).fit(X_tr, y_tr)
P_cal, P_te = clf.predict_proba(X_cal), clf.predict_proba(X_te)
s_cal = 1 - P_cal[np.arange(len(y_cal)), y_cal]


def sets_split(P, alpha):
    return (1 - P) <= conformal_quantile(s_cal, alpha)


def sets_mondrian(P, alpha):
    q = np.array([conformal_quantile(s_cal[y_cal == k], alpha) for k in (0, 1)])
    return (1 - P) <= q


def report(name, alert, y) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, alert, labels=[0, 1]).ravel()
    return {"regla": name, "recall": tp / (tp + fn), "precision": tp / max(tp + fp, 1),
            "alertas": int(alert.sum()), "pct_alertas": float(alert.mean()),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


ALPHA_C = 0.05
t_mondrian = 1 - conformal_quantile(s_cal[y_cal == 1], ALPHA_C)   # umbral equivalente sobre p(fraude)
S_split, S_mo = sets_split(P_te, ALPHA_C), sets_mondrian(P_te, ALPHA_C)
rules = pd.DataFrame([
    report("Umbral 0,5", P_te[:, 1] >= 0.5, y_te),
    report("Split conformal 95 %", S_split[:, 1], y_te),
    report("Mondrian 95 %", S_mo[:, 1], y_te),
])
results["fraude"] = {
    "n_train": len(y_tr), "n_cal": len(y_cal), "n_test": len(y_te), "fraudes_cal": int(y_cal.sum()),
    "tasa_fraude": float(yh.mean()), "scale_pos_weight": float(SCALE_POS_WEIGHT),
    "pr_auc_test": float(average_precision_score(y_te, P_te[:, 1])),
    "umbral_mondrian_95": t_mondrian,
    "umbral_split_95": float(1 - conformal_quantile(s_cal, ALPHA_C)),
    "q_unico_95": conformal_quantile(s_cal, ALPHA_C),
    "q_legitimas_95": conformal_quantile(s_cal[y_cal == 0], ALPHA_C),
    "q_fraudes_95": conformal_quantile(s_cal[y_cal == 1], ALPHA_C),
    "legitimas_cal": int((y_cal == 0).sum()),
    "cobertura_global_split_95": float(S_split[np.arange(len(y_te)), y_te].mean()),
    "cobertura_global_mondrian_95": float(S_mo[np.arange(len(y_te)), y_te].mean()),
    "vacios_mondrian_95": float((S_mo.sum(1) == 0).mean()),
    "reglas_test": rules.to_dict(orient="records"),
}

# Datos futuros
prod = pd.read_csv(DATA / "fraudTest.csv", usecols=COLS)
Xp, yp = features(prod), prod["is_fraud"].to_numpy()
for c in ("category", "gender"):
    Xp[c] = pd.Categorical(Xp[c], categories=Xh[c].cat.categories)
P_prod = clf.predict_proba(Xp)
rules_prod = pd.DataFrame([
    report("Umbral 0,5", P_prod[:, 1] >= 0.5, yp),
    report("Mondrian 95 %", sets_mondrian(P_prod, ALPHA_C)[:, 1], yp),
])
results["fraude"]["reglas_produccion"] = rules_prod.to_dict(orient="records")
results["fraude"]["produccion_tasa_fraude"] = float(yp.mean())

# Figura B1: scores de calibracion por clase y su umbral (la idea de Mondrian)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for ax, k, color, name in [(axes[0], 0, BLUE, "Legítimas"), (axes[1], 1, ORANGE, "Fraudes")]:
    sk = s_cal[y_cal == k]
    qk = conformal_quantile(sk, ALPHA_C)
    ax.hist(sk, bins=np.linspace(0, 1, 41), color=color, edgecolor=SURFACE, linewidth=1)
    ax.set_yscale("log")
    ax.axvline(qk, color=INK, lw=2, ls="--")
    qtxt = f"{qk:.3f}".replace(".", ",")
    ax.text(0.5, 0.95, f"umbral q̂ = {qtxt}\nel 95 % de los casos\nqueda a su izquierda", transform=ax.transAxes,
            ha="center", va="top", color=INK, fontsize=11,
            bbox=dict(boxstyle="round,pad=0.35", facecolor=SURFACE, edgecolor=GRID))
    ax.set_title(f"{name}: {len(sk):,} casos de calibración".replace(",", "."))
    ax.set(xlabel="error del modelo = 1 − probabilidad que le dio a la respuesta correcta",
           ylabel="número de casos (escala log)")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0\nacertó", "0,25", "0,5", "0,75", "1\nse equivocó"])
    ax.xaxis.label.set_size(10.5)
plt.tight_layout()
plt.savefig(OUT / "b_scores_mondrian.png", dpi=200)
plt.close()

# Figura B2: recall y precision por regla de decision (metricas del curso)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
colors = [MUTED, BLUE, ORANGE]
labels = ["Umbral 0,5\n(sin conformal)", "1. Split conformal\n95 % (umbral único)", "3. Mondrian 95 %\n(umbral por clase)"]
for ax, col, title in [(axes[0], "recall", "Recall: ¿cuántos fraudes detecto?"),
                       (axes[1], "precision", "Precisión: ¿cuántas alertas son fraude?")]:
    bars = ax.bar(labels, rules[col], color=colors, width=0.6)
    for r, v in zip(bars, rules[col]):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.02, f"{v:.1%}".replace(".", ","), ha="center", color=INK2, fontsize=11)
    ax.set(ylim=(0, 1.15), title=title)
    ax.yaxis.set_major_formatter(pct)
    ax.grid(axis="x", visible=False)
axes[0].axhline(0.95, color=INK, ls="--", lw=1.5, label="recall prometido 95 %")
axes[0].legend(loc="upper left")
plt.tight_layout()
plt.savefig(OUT / "b_reglas_recall_precision.png", dpi=200)
plt.close()

# Figura B3: recall garantizado vs. nivel pedido, y su precio en alertas
levels = np.linspace(0.80, 0.99, 20)
rec_mo = [sets_mondrian(P_te, 1 - l)[y_te == 1, 1].mean() for l in levels]
rec_sp = [sets_split(P_te, 1 - l)[y_te == 1, 1].mean() for l in levels]
alerts_mo = [sets_mondrian(P_te, 1 - l)[:, 1].mean() for l in levels]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
axes[0].plot(levels, levels, color=MUTED, ls="--", lw=1.5, label="ideal: recall = nivel pedido")
axes[0].plot(levels, rec_mo, color=ORANGE, lw=2, marker="o", ms=5, label="Mondrian")
axes[0].plot(levels, rec_sp, color=BLUE, lw=2, marker="o", ms=5, label="Split conformal (umbral único)")
axes[0].set(xlabel="nivel pedido 1 − α", ylabel="recall de fraude (test)", title="Mondrian cumple la garantía en todo α")
axes[0].yaxis.set_major_formatter(pct)
axes[0].xaxis.set_major_formatter(pct)
axes[0].legend(loc="lower right")
axes[1].plot(levels, np.array(alerts_mo) * 100, color=VIOLET, lw=2, marker="o", ms=5)
axes[1].set(xlabel="nivel pedido 1 − α", ylabel="% de transacciones alertadas",
            title="El precio de más recall: más alertas")
axes[1].xaxis.set_major_formatter(pct)
for ax in axes:
    ax.set_xticks([0.80, 0.85, 0.90, 0.95, 0.99])
plt.tight_layout()
plt.savefig(OUT / "b_recall_vs_alpha.png", dpi=200)
plt.close()
results["fraude"]["curva_mondrian"] = {f"{l:.2f}": {"recall": float(r), "alertas": float(a)}
                                       for l, r, a in zip(levels, rec_mo, alerts_mo)}

# Figura B4: matrices de confusion de Mondrian 95 %: test (mismo periodo) vs. produccion (futuro)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
for ax, r, title in [(axes[0], rules.iloc[2], "Datos de prueba (mismo periodo, 60.000)"),
                     (axes[1], rules_prod.iloc[1], "Datos futuros (meses siguientes, 555.719)")]:
    m = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
    ax.imshow(m / m.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1.3)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{m[i, j]:,}".replace(",", "."), ha="center", va="center", fontsize=15, weight="bold", color=INK)
    ax.set_xticks([0, 1], ["no alerta", "alerta"])
    ax.set_yticks([0, 1], ["legítima", "fraude"])
    ax.set(xlabel="decisión", ylabel="real")
    ax.set_title(f"{title}\nrecall {r['recall']:.1%} · precisión {r['precision']:.1%}")
    ax.grid(False)
plt.tight_layout()
plt.savefig(OUT / "b_matrices_confusion.png", dpi=200)
plt.close()

# =============================================================================
# PARTE C. Cuanto varia la cobertura segun n de calibracion (distribucion Beta)
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 4))
grid_c = np.linspace(0.8, 1.0, 600)
for n_cal, color in [(50, ORANGE), (500, BLUE), (5000, AQUA)]:
    k = int(np.ceil((n_cal + 1) * 0.9))
    dist = stats.beta(k, n_cal + 1 - k)
    ax.plot(grid_c, dist.pdf(grid_c), color=color, lw=2, label=f"n = {n_cal:,}".replace(",", "."))
    results.setdefault("variabilidad", {})[n_cal] = {"p05": float(dist.ppf(0.05)), "p95": float(dist.ppf(0.95))}
ax.axvline(0.9, color=INK, ls="--", lw=1.5)
ax.set(xlabel="cobertura obtenida en un despliegue", ylabel="densidad", yticks=[],
       title="Con pocos datos de calibración la cobertura real oscila")
ax.legend(title="calibración")
plt.tight_layout()
plt.savefig(OUT / "c_variabilidad_calibracion.png", dpi=200)
plt.close()

(OUT / "resultados.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
print(json.dumps(results, indent=2, ensure_ascii=False, default=float))
