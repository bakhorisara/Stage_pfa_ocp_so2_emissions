# %% [markdown]
# # Phase 1b -- Ameliorer le R2 en restant sur de la REGRESSION LINEAIRE
#
# Aucune methode non lineaire (pas d'arbre, pas de boosting). On compare
# uniquement des regressions lineaires, comme le demande le document :
#   - OLS (regression lineaire simple)
#   - Ridge  (lineaire + penalite L2)     <- attendu meilleur : features tres colineaires
#   - Lasso  (lineaire + penalite L1)
#   - ElasticNet (lineaire + L1 + L2)
# On fait aussi varier ce que le document autorise : le jeu de features
# (base 159  vs  enrichi) et le reglage de la validation glissante.
# Nettoyage, cible = variation, walk-forward + embargo, controles de fuite :
# INCHANGES.

# %%
import warnings, copy
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import so2lib as S

warnings.filterwarnings("ignore")
pd.set_option("display.width", 170); pd.set_option("display.max_columns", 30)
print = __import__("functools").partial(print, flush=True)

CFG = S.load_config("config.yaml")
OUT = Path("outputs"); FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
H = CFG["target"]["horizon_min"]
np.random.seed(CFG["random_seed"])

LINEAR_MODELS = ["linear", "ridge", "lasso", "elasticnet"]

df_raw = S.load_raw(CFG)
df_clean, _ = S.apply_cleaning(df_raw, CFG)
df_blk = S.make_blocks(df_clean, CFG)
print(f"blocs={df_blk['block_id'].nunique()}  lignes propres={len(df_blk)}  horizon={H} min")

# %% [markdown]
# ## A. Jeu de features : base (159) vs enrichi

# %%
CFG_BASE = copy.deepcopy(CFG)
CFG_BASE["features"].update(rolling_std_min=[], diff_min=[], delta_t_lags_min=[],
                            add_time_features=False)
feat_variants = {"base_159": CFG_BASE, "enrichi": CFG}
feats = {}
for name, cfg in feat_variants.items():
    dff = S.build_features(df_blk, cfg)
    y = S.build_target(dff, H, cfg)
    W = S.valid_windows(dff, y, cfg)
    feats[name] = (cfg, dff, W)
    print(f"  {name:10s}: {len(S.feature_columns(cfg)):4d} features, {len(W)} fenetres valides (h={H})")

# %% [markdown]
# ## B. Comparaison des regressions lineaires (walk-forward + embargo)
#
# On rapporte le R2 moyen +/- ecart entre replis (comme le document) ET le R2
# poole (tous les points test concatenes -- moins sensible a un petit repli faible).

# %%
emb = int(CFG["features"]["lookback_min"] + H)
rows = []
for fname, (cfg, dff, W) in feats.items():
    for kind in LINEAR_MODELS:
        folds, pooled = S.walk_forward_pooled(W, cfg, kind=kind, embargo_min=emb)
        s = S.summarize_folds(folds)
        rows.append({"features": fname, "model": kind,
                     "r2_mean": s["r2_mean"], "r2_std": s["r2_std"],
                     "r2_pooled": pooled["pooled_r2"],
                     "mae_mean": s["mae_mean"], "naive_mae": s["naive_mae_mean"],
                     "mae_gain_%": s["mae_gain_pct"],
                     "beats_naive_all_folds": s["beats_naive_all_folds"]})
        print(f"  {fname:9s} {kind:11s} R2_moy={s['r2_mean']:+.3f}±{s['r2_std']:.3f}  "
              f"R2_poole={pooled['pooled_r2']:+.3f}  "
              f"MAE {s['mae_mean']:.1f} vs naive {s['naive_mae_mean']:.1f} ({s['mae_gain_pct']:+.1f}%)  "
              f"bat_naive_tous_replis={s['beats_naive_all_folds']}")
res = pd.DataFrame(rows).sort_values("r2_pooled", ascending=False)
res.to_csv(OUT / "phase1b_comparaison_modeles.csv", index=False)
print("\n=== classement (R2 poole) ===")
print(res.to_string(index=False))

# %% [markdown]
# ## C. Reglage de la validation glissante pour le meilleur couple (features, modele)

# %%
best = res.iloc[0]
best_feat, best_model = best["features"], best["model"]
cfg_b, dff_b, W_b = feats[best_feat]
print(f"meilleur couple : features='{best_feat}', modele='{best_model}'")

tune_rows = []
for mtf in [0.35, 0.45, 0.55, 0.65]:
    for nf in [4, 5, 6]:
        cfg_t = copy.deepcopy(cfg_b)
        cfg_t["validation"].update(min_train_frac=mtf, n_folds=nf)
        folds, pooled = S.walk_forward_pooled(W_b, cfg_t, kind=best_model, embargo_min=emb)
        s = S.summarize_folds(folds)
        tune_rows.append({"min_train_frac": mtf, "n_folds": nf,
                          "r2_mean": s["r2_mean"], "r2_std": s["r2_std"],
                          "r2_pooled": pooled["pooled_r2"], "mae_gain_%": s["mae_gain_pct"],
                          "beats_all": s["beats_naive_all_folds"]})
        print(f"  min_train={mtf:.2f} folds={nf}: R2_moy={s['r2_mean']:+.3f}±{s['r2_std']:.3f} "
              f"R2_poole={pooled['pooled_r2']:+.3f}  bat_naive_tous={s['beats_naive_all_folds']}")
tune = pd.DataFrame(tune_rows).sort_values("r2_pooled", ascending=False)
tune.to_csv(OUT / "phase1b_reglage_validation.csv", index=False)
print("\n", tune.to_string(index=False))

# %% [markdown]
# ## D. Balayage d'horizon avec le meilleur modele lineaire

# %%
sweep_rows = []
for h in CFG["target"]["horizon_sweep_min"]:
    y = S.build_target(dff_b, h, cfg_b)
    W = S.valid_windows(dff_b, y, cfg_b)
    folds, pooled = S.walk_forward_pooled(W, cfg_b, kind=best_model,
                                          embargo_min=int(CFG["features"]["lookback_min"] + h))
    s = S.summarize_folds(folds)
    sweep_rows.append({"horizon_min": h, "n_windows": len(W), "r2_mean": s["r2_mean"],
                       "r2_std": s["r2_std"], "r2_pooled": pooled["pooled_r2"],
                       "mae_mean": s["mae_mean"], "naive_mae": s["naive_mae_mean"],
                       "mae_gain_%": s["mae_gain_pct"], "beats_all": s["beats_naive_all_folds"]})
    print(f"  h={h:3d}: R2_moy={s['r2_mean']:+.3f}±{s['r2_std']:.3f} R2_poole={pooled['pooled_r2']:+.3f} "
          f"MAE {s['mae_mean']:.1f} vs {s['naive_mae_mean']:.1f} ({s['mae_gain_pct']:+.1f}%)")
sweep = pd.DataFrame(sweep_rows)
sweep.to_csv(OUT / "phase1b_balayage_horizon.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(sweep.horizon_min, sweep.r2_mean, yerr=sweep.r2_std, marker="o", capsize=3,
            label=f"{best_model} : R2 moy ± ecart replis")
ax.plot(sweep.horizon_min, sweep.r2_pooled, marker="s", ls="--", label=f"{best_model} : R2 poole")
ax.axhline(0, color="grey", lw=.8)
ax.set_xlabel("horizon (min)"); ax.set_ylabel("R2")
ax.set_title(f"R2 vs horizon -- features '{best_feat}', modele lineaire '{best_model}'")
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(FIG / "phase1b_r2_par_horizon.png", dpi=120); plt.close(fig)

# %% [markdown]
# ## E. Le gagnant repasse les controles de fuite (4 protocoles + controle negatif)

# %%
CFG_WIN = copy.deepcopy(cfg_b)
bt = tune.iloc[0]
CFG_WIN["validation"].update(min_train_frac=float(bt["min_train_frac"]), n_folds=int(bt["n_folds"]))
y = S.build_target(dff_b, H, CFG_WIN); W = S.valid_windows(dff_b, y, CFG_WIN)
cols = S.feature_columns(CFG_WIN)
X = W[cols].to_numpy(float); yy = W["y"].to_numpy(float); n = len(W)
date = W[CFG_WIN["data"]["date_col"]].to_numpy()
frac = CFG_WIN["leakage"]["test_frac"]; cut = int(n * (1 - frac))

from sklearn.utils import shuffle as sk_shuffle
prot = {}
Xs, ys = sk_shuffle(X, yy, random_state=0)
prot["A_shuffle"] = S.metrics(ys[cut:], S._fit_predict(Xs[:cut], ys[:cut], Xs[cut:], best_model, CFG_WIN)[0])["r2"]
prot["B_chrono"] = S.metrics(yy[cut:], S._fit_predict(X[:cut], yy[:cut], X[cut:], best_model, CFG_WIN)[0])["r2"]
cutoff = pd.Timestamp(date[cut]) - pd.Timedelta(minutes=emb)
tr = np.where(date <= np.datetime64(cutoff))[0]
prot["C_chrono_embargo"] = S.metrics(yy[cut:], S._fit_predict(X[tr], yy[tr], X[cut:], best_model, CFG_WIN)[0])["r2"]
folds, pooled = S.walk_forward_pooled(W, CFG_WIN, kind=best_model, embargo_min=emb)
prot["D_walkforward_mean"] = S.summarize_folds(folds)["r2_mean"]
prot["D_walkforward_pooled"] = pooled["pooled_r2"]
print("4 protocoles (R2) :", {k: round(v, 4) for k, v in prot.items()})
flagA = (prot["A_shuffle"] - prot["D_walkforward_mean"]) > 0.15
print(f"  A domine D ? {flagA}  (doit etre False)")

negs = []
for rep in range(3):
    rng = np.random.default_rng(rep)
    yperm = rng.permutation(yy)
    negs.append(S.metrics(yperm[cut:], S._fit_predict(X[tr], yperm[tr], X[cut:], best_model, CFG_WIN)[0])["r2"])
print(f"controle negatif R2 = {np.mean(negs):+.5f}  {[round(v, 5) for v in negs]}  (doit etre ~0)")

# %%
verdict = {
    "modele": f"regression lineaire ({best_model})", "features": best_feat,
    "reglage_validation": {"min_train_frac": float(bt["min_train_frac"]), "n_folds": int(bt["n_folds"])},
    "R2_moyen_replis": float(bt["r2_mean"]), "R2_ecart_replis": float(bt["r2_std"]),
    "R2_poole": float(bt["r2_pooled"]), "MAE_gain_pct": float(bt["mae_gain_%"]),
    "fuite_4protocoles": {k: float(v) for k, v in prot.items()},
    "A_domine_D": bool(flagA), "controle_negatif_R2": float(np.mean(negs)),
}
S.save_json(verdict, OUT / "phase1b_verdict.json")
print("\n================= VERDICT PHASE 1b (lineaire seulement) =================")
for k, v in verdict.items():
    print(f"  {k:20s} {v}")
print("=======================================================================")
