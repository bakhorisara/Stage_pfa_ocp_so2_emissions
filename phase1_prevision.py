# %% [markdown]
# # Phase 1 -- Prevision de la variation des emissions SO2
#
# Fichier : defini dans `config.yaml` (par defaut donnees synthetiques).
# Toute la logique est dans `so2lib.py`,
# tous les parametres dans `config.yaml`. Chaque etape affiche un **checkpoint**
# (valeur calculee vs repere du document).

# %%
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import so2lib as S

warnings.filterwarnings("ignore")
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 40)

CFG = S.load_config("config.yaml")
OUT = Path("outputs")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
CK = []          # liste de checkpoints -> JSON final
np.random.seed(CFG["random_seed"])


def show_fig(fig, path):
    """Enregistre le PNG (pour le rapport LaTeX) ET l'affiche dans le notebook."""
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    try:
        from IPython.display import Image, display
        display(Image(filename=str(path)))
    except Exception:
        pass

# %% [markdown]
# ## 1. Chargement + audit des donnees brutes

# %%
df_raw = S.load_raw(CFG)
aud = S.audit(df_raw, CFG)
print(f"shape .................. {aud['shape']}")
print(f"plage temporelle ...... {aud['date_min']}  ->  {aud['date_max']}")
print(f"horodatages dupliques . {aud['dup_timestamps']}")
print(f"trous > 1 min (brut) .. {aud['raw_gaps_gt_1min']}")
print("NaN par colonne :")
for c, v in aud["nan_per_col"].items():
    if v:
        print(f"    {c:22s} {v}")
print("\nvaleurs les plus repetees / negatifs par colonne :")
for c, st in aud["col_stats"].items():
    print(f"    {c:22s} min={st['min']:12.3f} max={st['max']:12.3f} "
          f"top={st['top_value']} x{st['top_count']:<6d} neg={st['n_negative']}")

CK.append(S.checkpoint("Lignes chargees", aud["shape"][0], 35260, tol=0))
CK.append(S.checkpoint("Horodatages dupliques", aud["dup_timestamps"], 0, tol=0))
S.save_json(aud, OUT / "audit_brut.json")

# %% [markdown]
# ## 2. Decouverte des codes defaut deguises en mesures
#
# Regle : une valeur repetee >= `min_repeats` fois **et** qui viole une borne
# physique evidente est un code defaut. Comparaison `np.isclose` (piege float32).

# %%
faults = S.discover_fault_codes(df_raw, CFG)
print("Codes defaut DECOUVERTS (col -> [(valeur, occurrences, %)]) :")
for c, hits in faults.items():
    for v, n, pct in hits:
        print(f"    {c:22s} {v:>16}   x{n:<6d} ({pct} %)")
if not faults:
    print("    (aucun via la decouverte auto ; la liste connue reste appliquee)")
print("\nRappel : INPUT_SULF_FLOW ~ -0.00325 (x442) est un near-zero PLAUSIBLE "
      "(ligne au ralenti) -> NON retenu comme code defaut.")
CK.append(S.checkpoint("Colonnes avec code defaut decouvert", len(faults)))
S.save_json(faults, OUT / "codes_defaut_decouverts.json")

# %% [markdown]
# ## 3. Nettoyage -- sentinelles + bornes physiques -> NaN, puis dropna()
#
# Aucune imputation / interpolation / clip. Regle unique : toute ligne avec un
# NaN dans une colonne est supprimee.

# %%
df_clean, clean_rep = S.apply_cleaning(df_raw, CFG)
print(f"lignes en entree .............. {clean_rep['rows_in']}")
print(f"doublons horodatage retires .. {clean_rep.get('dup_dropped', 0)}")
print(f"NaN ajoutes (sentinelles) .... {clean_rep['sentinel_nan_added']}")
print(f"NaN ajoutes (bornes phys.) ... {clean_rep['bound_nan_added']}")
print(f"suppression dropna() SEUL .... {clean_rep['drop_rate_dropna_only_pct']} %")
print(f"suppression TOTALE ........... {clean_rep['drop_rate_total_pct']} %")
print(f"lignes conservees ............ {clean_rep['rows_out']}  ({clean_rep['kept_pct']} %)")

CK.append(S.checkpoint("Taux suppression total (%)", clean_rep["drop_rate_total_pct"], 7.78, tol=1.0, pct=True))
CK.append(S.checkpoint("Part conservee (%)", clean_rep["kept_pct"], 92.22, tol=1.0, pct=True))
S.save_json(clean_rep, OUT / "nettoyage_rapport.json")

# %% [markdown]
# ## 4. Blocs temporels continus
#
# Une fois les lignes invalides retirees, la chronologie se scinde en segments
# de minutes consecutives. Tout lag / moyenne mobile se calcule DANS un bloc.

# %%
df_blk = S.make_blocks(df_clean, CFG)
bst = S.block_stats(df_blk, CFG)
for k, v in bst.items():
    print(f"    {k:20s} {v}")
CK.append(S.checkpoint("Blocs contigus", bst["n_blocks"], 134, tol=20))

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df_blk.groupby("block_id").size(), bins=60, color="#3b7ea1")
ax.set_xlabel("longueur du bloc (minutes)"); ax.set_ylabel("nb de blocs")
ax.set_title("Distribution des longueurs de blocs")
fig.tight_layout(); show_fig(fig, FIG / "blocs_longueurs.png")

# %% [markdown]
# ## 5. Grille de faisabilite -- choix de la fenetre de lookback
#
# Pour chaque (lookback, horizon) : nb de fenetres exploitables et part
# concentree sur les 10 derniers jours. On cherche la rupture nette.

# %%
grid = S.feasibility_grid(df_blk, CFG)
print(grid.to_string(index=False))
grid.to_csv(OUT / "grille_faisabilite.csv", index=False)

piv = grid.pivot(index="lookback_min", columns="horizon_min", values="n_windows")
print("\nFenetres exploitables (lignes = lookback, colonnes = horizon) :")
print(piv.to_string())
rec_col = [c for c in grid.columns if c.startswith("pct_in_last_")][0]
piv2 = grid.pivot(index="lookback_min", columns="horizon_min", values=rec_col)
print(f"\n{rec_col} :")
print(piv2.to_string())
print(f"\n-> lookback retenu (config.features.lookback_min) : {CFG['features']['lookback_min']} min")

# %% [markdown]
# ## 6. Construction des features (Delta_T + lags + moyennes mobiles, PAR BLOC)

# %%
df_feat = S.build_features(df_blk, CFG)
feat_cols = S.feature_columns(CFG)
f = CFG["features"]
enrichi = bool(f.get("rolling_std_min") or f.get("diff_min") or f.get("delta_t_lags_min")
               or f.get("add_time_features"))
print(f"nombre de features : {len(feat_cols)}  ({'jeu ENRICHI' if enrichi else 'jeu de base'})")
print("exemples :", feat_cols[:6], "...", feat_cols[-4:])
# repere doc = 166 pour le jeu de base ; l'enrichissement (volatilite, momentum,
# lags Delta_T, saison) est un choix explicite valide en Phase 1b.
CK.append(S.checkpoint("Nombre de variables", len(feat_cols),
                       None if enrichi else 166, tol=None if enrichi else 40))

# verif anti-fuite : une feature de lag doit etre NaN en tete de chaque bloc
chk = df_feat.groupby("block_id").head(1)[f"{CFG['data']['raw_channels'][0]}_lag1"]
print(f"lag1 NaN en tete de bloc : {chk.isna().all()}  (doit etre True)")

# %% [markdown]
# ## 7. Cible = variation, et fenetres valides (a l'horizon retenu)

# %%
H = CFG["target"]["horizon_min"]
y_H = S.build_target(df_feat, H, CFG)
W_H = S.valid_windows(df_feat, y_H, CFG)
print(f"horizon = {H} min")
print(f"fenetres valides : {len(W_H)}")
print(f"y : mean={W_H['y'].mean():.3f}  std={W_H['y'].std():.3f}  "
      f"min={W_H['y'].min():.1f}  max={W_H['y'].max():.1f}")
nm = S.naive_metrics(W_H["y"])
print(f"reference naive (delta=0) : MAE={nm['mae']:.3f}  RMSE={nm['rmse']:.3f}  R2={nm['r2']:.4f}")
CK.append(S.checkpoint("Fenetres valides (h=%d)" % H, len(W_H)))
CK.append(S.checkpoint("MAE reference naive (h=%d)" % H, round(nm["mae"], 3)))

# %% [markdown]
# ## 8. Modeles : regression lineaire simple ET Lasso (walk-forward)

# %%
print(">>> Regression lineaire")
fold_lin = S.run_walk_forward(W_H, CFG, kind="linear", verbose=True)
sum_lin = S.summarize_folds(fold_lin)
print(">>> Lasso (alpha via TimeSeriesSplit)")
fold_las = S.run_walk_forward(W_H, CFG, kind="lasso", verbose=True)
sum_las = S.summarize_folds(fold_las)

print("\n--- Regression lineaire ---")
for k, v in sum_lin.items():
    print(f"    {k:22s} {v}")
print("--- Lasso ---")
for k, v in sum_las.items():
    print(f"    {k:22s} {v}")
if "alpha" in fold_las.columns:
    print("    alphas par repli :", [round(a, 5) for a in fold_las["alpha"].tolist()])
    print("    coefs non nuls   :", fold_las["n_nonzero"].tolist())

winner = "lineaire" if sum_lin["r2_mean"] >= sum_las["r2_mean"] else "lasso"
winner_kind = "linear" if winner == "lineaire" else "lasso"
sum_win = sum_lin if winner == "lineaire" else sum_las
print(f"\n-> modele retenu : REGRESSION {winner.upper()}  "
      f"(R2 {sum_win['r2_mean']:.3f} +/- {sum_win['r2_std']:.3f}, "
      f"MAE {sum_win['mae_mean']:.1f} vs naive {sum_win['naive_mae_mean']:.1f})")
print("   NB : OLS non regularisee est instable avec autant de features colineaires ;")
print("        Lasso (L1) reste une REGRESSION LINEAIRE et gere la colinearite.")
fold_lin.to_csv(OUT / "folds_lineaire_hH.csv", index=False)
fold_las.to_csv(OUT / "folds_lasso_hH.csv", index=False)
CK.append(S.checkpoint("R2 OLS (h=%d, walk-forward)" % H, round(sum_lin["r2_mean"], 4)))
CK.append(S.checkpoint("R2 Lasso (h=%d, walk-forward)" % H, round(sum_las["r2_mean"], 4)))
CK.append(S.checkpoint("R2 modele retenu (h=%d)" % H, round(sum_win["r2_mean"], 4), 0.5735, tol=0.20))

# %% [markdown]
# ## 9. Balayage d'horizon (5 -> 55 min)
#
# On verifie si le R2 croit de facon monotone (effet retour-a-la-moyenne).
# Le balayage utilise le Lasso a alpha FIGE (= alpha choisi par TimeSeriesSplit
# sur le probleme retenu h=55, identique a chaque repli) pour rester rapide ;
# le resultat retenu h=55 (section 8) garde la selection complete par CV.

# %%
sweep_kind = "lasso_fixed" if winner_kind == "lasso" else winner_kind
sweep = S.horizon_sweep(df_feat, CFG, kind=sweep_kind)
print(sweep.to_string(index=False))
sweep.to_csv(OUT / "balayage_horizon.csv", index=False)

fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.errorbar(sweep["horizon_min"], sweep["r2_mean"], yerr=sweep["r2_std"],
             marker="o", capsize=3, color="#1f77b4", label="R2 (moy +/- ecart replis)")
ax1.axhline(0, color="grey", lw=.8)
ax1.set_xlabel("horizon (minutes)"); ax1.set_ylabel("R2")
ax1.set_title(f"R2 du modele retenu ({winner_kind}) selon l'horizon de prevision")
ax2 = ax1.twinx()
ax2.plot(sweep["horizon_min"], sweep["mae_mean"], marker="s", color="#d62728", label="MAE modele")
ax2.plot(sweep["horizon_min"], sweep["naive_mae_mean"], marker="x", ls="--",
         color="#d62728", alpha=.6, label="MAE naive")
ax2.set_ylabel("MAE")
l1, la1 = ax1.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2, la1 + la2, loc="upper left", fontsize=8)
fig.tight_layout(); show_fig(fig, FIG / "r2_par_horizon.png")

mono = bool((sweep["r2_mean"].diff().dropna() >= -0.02).all())
CK.append(S.checkpoint("R2 croissant (quasi) monotone avec l'horizon", mono))
best_h = int(sweep.loc[sweep["r2_mean"].idxmax(), "horizon_min"])
CK.append(S.checkpoint("Horizon de R2 maximal", best_h, 55, tol=15))

# %% [markdown]
# ## 10. Verification de fuite (1) -- 4 protocoles de decoupage

# %%
prot = S.leakage_protocols(W_H, CFG, kind=winner_kind)
print(prot.to_string(index=False))
prot.to_csv(OUT / "fuite_4protocoles.csv", index=False)

r2A = float(prot.loc[prot.protocol == "A_shuffle_8020", "r2"].iloc[0])
r2D = float(prot.loc[prot.protocol == "D_walkforward_embargo", "r2"].iloc[0])
flag_A = (r2A - r2D) > 0.15
print(f"\nR2(A melange) = {r2A:.4f}   R2(D glissant+embargo) = {r2D:.4f}   "
      f"ecart = {r2A - r2D:+.4f}")
print("  -> " + ("ALERTE : A domine nettement -> fuite par chevauchement temporel probable"
                  if flag_A else "pas de domination anormale de A -- coherent"))

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(prot["protocol"], prot["r2"], color=["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"])
ax.set_ylabel("R2"); ax.set_title("R2 selon le protocole de decoupage")
ax.tick_params(axis="x", rotation=20)
fig.tight_layout(); show_fig(fig, FIG / "fuite_4protocoles.png")
CK.append(S.checkpoint("Protocole A ne domine pas D", (not flag_A)))

# %% [markdown]
# ## 11. Verification de fuite (2) -- controle negatif (cible permutee seule)
#
# Les features restent a leur vraie position, seule la cible est melangee.
# Tout R2 positif ici est la preuve d'une fuite ailleurs dans le pipeline.

# %%
negc = S.negative_control(W_H, CFG, kind=winner_kind)
print(f"R2 controle negatif : {negc['neg_control_r2_mean']:.5f}   "
      f"(replis : {[round(v, 5) for v in negc['neg_control_r2_all']]})")
ok_neg = negc["neg_control_r2_mean"] < 0.02
print("  -> " + ("OK : ~0, pas de fuite detectee par ce test"
                  if ok_neg else "ALERTE : R2 positif -> chercher la fuite AVANT de continuer"))
CK.append(S.checkpoint("Controle negatif R2 ~ 0", round(negc["neg_control_r2_mean"], 5), 0.0, tol=0.02))

# %% [markdown]
# ## 12. Sauvegarde du modele retenu + tableau de controle

# %%
import joblib
mdl_final, scaler_final, cols_final = S.fit_final_model(W_H, CFG, kind=winner_kind)
joblib.dump({"model": mdl_final, "scaler": scaler_final, "feature_cols": cols_final,
             "model_kind": winner_kind, "horizon_min": H, "config": CFG},
            OUT / f"model_retenu_h{H}.pkl")
try:
    W_H.to_parquet(OUT / f"fenetres_valides_h{H}.parquet")
except Exception as _e:
    W_H.to_csv(OUT / f"fenetres_valides_h{H}.csv", index=False)

summary = {
    "fichier": CFG["data"]["csv_path"],
    "horizon_retenu_min": H,
    "lookback_min": CFG["features"]["lookback_min"],
    "n_features": len(feat_cols),
    "n_blocs": bst["n_blocks"],
    "taux_suppression_pct": clean_rep["drop_rate_total_pct"],
    "n_fenetres_valides": len(W_H),
    "ols": sum_lin, "lasso": sum_las,
    "modele_retenu": winner, "modele_retenu_kind": winner_kind,
    "R2_retenu_mean": sum_win["r2_mean"], "R2_retenu_std": sum_win["r2_std"],
    "naive": nm,
    "fuite_4protocoles": prot.to_dict(orient="records"),
    "controle_negatif": negc,
    "balayage_horizon": sweep.to_dict(orient="records"),
}
S.save_json(summary, OUT / "phase1_resume.json")
S.save_json(CK, OUT / "checkpoints_phase1.json")

print("\n================= TABLEAU DE CONTROLE -- PHASE 1 =================")
for c in CK:
    exp = "" if c["expected"] is None else f"  | repere doc : {c['expected']}"
    print(f"  [{c['status']:5s}] {c['name']:44s} = {c['value']}{exp}")
print("================================================================")
print("\nFichiers ecrits dans outputs/ :")
for p in sorted(OUT.glob("*")):
    if p.is_file():
        print("   ", p.name)
