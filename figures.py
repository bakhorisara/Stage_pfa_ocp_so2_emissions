# %% [markdown]
# # Figures -- regeneration centralisee
#
# Ce notebook reconstruit **toutes** les figures du projet a partir des resultats
# deja calcules (`outputs/*.csv`, `outputs/*.json`, `outputs/*.parquet`) et des
# donnees nettoyees. Il ne relance **pas** la modelisation ; execution < 1 min.
# Chaque figure est affichee ici et re-enregistree dans `outputs/figures/`.

# %%
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import Image, display
import so2lib as S

CFG = S.load_config("config.yaml")
OUT = Path("outputs"); FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
H = CFG["target"]["horizon_min"]


def show(fig, name):
    p = FIG / name
    fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    display(Image(filename=str(p)))
    print("->", p)

# %% [markdown]
# ## Données nettoyées (nécessaires pour 2 figures)

# %%
df_raw = S.load_raw(CFG)
df_clean, _ = S.apply_cleaning(df_raw, CFG)
df_blk = S.make_blocks(df_clean, CFG)
print(f"{len(df_blk)} lignes propres, {df_blk['block_id'].nunique()} blocs")

# %% [markdown]
# ## Figure 1 — Distribution des longueurs de blocs (Phase 1)

# %%
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df_blk.groupby("block_id").size(), bins=60, color="#3b7ea1")
ax.set_xlabel("longueur du bloc (minutes)"); ax.set_ylabel("nb de blocs")
ax.set_title("Distribution des longueurs de blocs")
fig.tight_layout(); show(fig, "blocs_longueurs.png")

# %% [markdown]
# ## Figure 2 — R² et MAE selon l'horizon de prévision (Phase 1)

# %%
sw = pd.read_csv(OUT / "balayage_horizon.csv")
fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.errorbar(sw["horizon_min"], sw["r2_mean"], yerr=sw["r2_std"], marker="o", capsize=3,
             color="#1f77b4", label="R2 (moy +/- ecart replis)")
ax1.axhline(0, color="grey", lw=.8)
ax1.set_xlabel("horizon (minutes)"); ax1.set_ylabel("R2")
ax1.set_title("R2 du modele retenu (Lasso) selon l'horizon")
ax2 = ax1.twinx()
ax2.plot(sw["horizon_min"], sw["mae_mean"], marker="s", color="#d62728", label="MAE modele")
ax2.plot(sw["horizon_min"], sw["naive_mae_mean"], marker="x", ls="--", color="#d62728",
         alpha=.6, label="MAE naive")
ax2.set_ylabel("MAE")
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
fig.tight_layout(); show(fig, "r2_par_horizon.png")

# %% [markdown]
# ## Figure 3 — R² selon le protocole de découpage (contrôle de fuite, Phase 1)

# %%
pr = pd.read_csv(OUT / "fuite_4protocoles.csv")
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(pr["protocol"], pr["r2"], color=["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"])
ax.set_ylabel("R2"); ax.set_title("R2 selon le protocole de decoupage")
ax.tick_params(axis="x", rotation=20)
for x, v in zip(range(len(pr)), pr["r2"]):
    ax.text(x, v + .01, f"{v:.3f}", ha="center", fontsize=8)
fig.tight_layout(); show(fig, "fuite_4protocoles.png")

# %% [markdown]
# ## Figure 4 — Sensibilité des leviers : 3 mesures indépendantes (Phase 2)

# %%
tab = pd.read_csv(OUT / "phase2_sensibilite_leviers.csv")
def slog(v):
    return np.sign(v) * np.log10(np.abs(v) + 1e-9)
fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(tab)); w = 0.27
ax.bar(x - w, slog(tab["analytic_per_unit"]), w, label="analytique (soutenu)")
ax.bar(x, slog(tab["empirical_sustained_per_unit"]), w, label="empirique soutenue")
ax.bar(x + w, slog(tab["empirical_calendar_per_unit"]), w, label="empirique calendrier (60 min)")
ax.axhline(0, color="grey", lw=.8); ax.set_xticks(x)
ax.set_xticklabels(tab["sensor"], rotation=25, ha="right", fontsize=8)
ax.set_ylabel("sign(.) * log10|.|")
ax.set_title("Sensibilite SO2 par levier : 3 mesures independantes")
ax.legend(fontsize=8)
fig.tight_layout(); show(fig, "phase2_sensibilite.png")

# %% [markdown]
# ## Figure 5 — Distributions des capteurs candidats (test de bimodalité, Phase 2)

# %%
bim = pd.read_csv(OUT / "phase2_bimodalite.csv")
cands = CFG["phase2"]["lever_candidates"]
ncol = 4; nrow = -(-len(cands) // ncol)          # grille : 4 colonnes (lisible)
fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.8 * nrow))
axes = np.atleast_1d(axes).ravel()
for ax in axes[len(cands):]:
    ax.axis("off")
for ax, s in zip(axes, cands):
    ax.hist(pd.to_numeric(df_blk[s], errors="coerce").dropna(), bins=80, color="#4573b3")
    flag = bool(bim.loc[bim.sensor == s, "is_bimodal"].iloc[0])
    ax.set_title(f"{s}\n{'BIMODAL' if flag else 'unimodal'}", fontsize=8)
    ax.tick_params(labelsize=7)
fig.tight_layout(); show(fig, "phase2_distributions.png")

# %% [markdown]
# ## Figure 6 — Trajectoire SO₂ observée vs seuil P95 (contrôleur MPC, Phase 2)

# %%
try:
    W = pd.read_parquet(OUT / f"fenetres_valides_h{H}.parquet")
except Exception:
    W = pd.read_csv(OUT / f"fenetres_valides_h{H}.csv", parse_dates=[CFG["data"]["date_col"]])
v = json.load(open(OUT / "phase2_verdict.json", encoding="utf-8"))
thr = v["seuils"]["95"]
cut = int(len(W) * 0.80)
W_sim = W.iloc[cut:].reset_index(drop=True)
real_future = W_sim[CFG["data"]["target_col"]].to_numpy() + W_sim["y"].to_numpy()
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(real_future, lw=.6, label="SO2(t+h) reel (sans commande)")
ax.axhline(thr, color="red", ls="--", lw=1, label=f"seuil P95 = {thr:.0f} ppm")
ax.set_title("Reference realisee vs seuil P95 -- periode de simulation du controleur")
ax.set_xlabel("pas de decision"); ax.set_ylabel("SO2 (ppm)"); ax.legend(fontsize=8)
fig.tight_layout(); show(fig, "phase2_trajectoire_P95.png")

# %% [markdown]
# ## Figure 7 — Comparaison des modèles linéaires (Phase 1b, si disponible)

# %%
f_cmp = OUT / "phase1b_comparaison_modeles.csv"
if f_cmp.exists():
    cmp = pd.read_csv(f_cmp).sort_values("r2_pooled")
    lab = cmp["features"] + " / " + cmp["model"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(lab, cmp["r2_mean"], xerr=cmp["r2_std"], color="#5b8def", capsize=3)
    ax.set_xlabel("R2 (moyenne des replis, barres = ecart-type)")
    ax.set_title("Phase 1b -- regressions lineaires comparees (h=55)")
    for y, (m, p) in enumerate(zip(cmp["r2_mean"], cmp["r2_pooled"])):
        ax.text(m + .01, y, f"{m:.3f} (pool {p:.3f})", va="center", fontsize=7)
    fig.tight_layout(); show(fig, "phase1b_comparaison_modeles.png")
else:
    print("phase1b_comparaison_modeles.csv absent -- lancer phase1b_modeles.py d'abord")

# %%
print("Toutes les figures regenerees dans outputs/figures/.")
