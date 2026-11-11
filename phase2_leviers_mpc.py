# %% [markdown]
# # Phase 2 -- Validation des leviers de commande + controleur MPC simule
#
# On repart du modele RETENU en Phase 1 (regression lineaire Lasso, enrichie,
# horizon 55 min). Rien n'est re-nettoye : meme pipeline, memes fenetres valides.
#
# Etapes :
#  1. Pour chaque capteur candidat : sensibilite ANALYTIQUE (somme des coefficients)
#     et EMPIRIQUE (decalage des 60 dernieres minutes d'historique, recalcul de
#     toutes les features derivees). Un levier n'est retenu que si les deux
#     mesures s'accordent sur le SIGNE (et, pour le controleur, sur l'ordre de grandeur).
#  2. Test de BIMODALITE sur chaque candidat : un capteur bimodal (equipement
#     parfois a l'arret) est ecarte du controleur.
#  3. Bornes / reglage habituel / vitesse max : MESURES sur les donnees.
#  4. Controleur MPC simule (cout 3 termes). Reference "sans commande" = la
#     trajectoire SO2 REELLEMENT observee ; le modele n'apporte que l'effet
#     incremental valide d'un ajustement de levier.

# %%
import warnings, json
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
CK = []


def show_fig(fig, path):
    """Enregistre le PNG (pour le rapport LaTeX) ET l'affiche dans le notebook."""
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    try:
        from IPython.display import Image, display
        display(Image(filename=str(path)))
    except Exception:
        pass

# --- pipeline commun (identique Phase 1) ------------------------------------
df_raw = S.load_raw(CFG)
df_clean, _ = S.apply_cleaning(df_raw, CFG)
df_blk = S.make_blocks(df_clean, CFG)
df_feat = S.build_features(df_blk, CFG)
y_H = S.build_target(df_feat, H, CFG)
W_H = S.valid_windows(df_feat, y_H, CFG)
print(f"blocs={df_blk['block_id'].nunique()}  fenetres valides (h={H})={len(W_H)}")

mdl, scaler, cols = S.fit_final_model(W_H, CFG, kind="lasso")
print(f"modele : Lasso  |  {int(np.sum(np.abs(mdl.coef_) > 1e-12))} coefs non nuls / {len(cols)}")

# %% [markdown]
# ## 1. Sensibilite des leviers : analytique vs empirique

# %%
tab = S.sign_agreement_table(df_feat, W_H, mdl, scaler, CFG)
pd.set_option("display.float_format", lambda v: f"{v:.4g}")
print(tab.to_string(index=False))
tab.to_csv(OUT / "phase2_sensibilite_leviers.csv", index=False)

for _, r in tab.iterrows():
    if not r["model_uses_sensor"]:
        tag = "CAPTEUR IGNORE PAR LE MODELE -> rejete"
    elif r["retained_for_controller"]:
        tag = "RETENU (signe + amplitude coherents)"
    elif r["sign_agree"]:
        tag = "signe OK mais amplitude douteuse"
    else:
        tag = "SIGNE EN DESACCORD (effet de calendrier) -> rejete"
    print(f"  {r['sensor']:20s} analytique={r['analytic_per_unit']:+.4g}  "
          f"emp.soutenue={r['empirical_sustained_per_unit']:+.4g}  "
          f"emp.calendrier={r['empirical_calendar_per_unit']:+.4g}  "
          f"ratio(analytic/soutenue)={r['amplitude_ratio_analytic_vs_sustained']:.2f}  -> {tag}")
CK.append(S.checkpoint("Leviers avec accord de signe (3 mesures)", int(tab["sign_agree"].sum())))
CK.append(S.checkpoint("Leviers retenus pour le controleur", int(tab["retained_for_controller"].sum())))

fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(tab)); w = 0.27
def _slog(v):
    return np.sign(v) * np.log10(np.abs(v) + 1e-9)
ax.bar(x - w, _slog(tab["analytic_per_unit"]), w, label="analytique (soutenu)")
ax.bar(x, _slog(tab["empirical_sustained_per_unit"]), w, label="empirique soutenue")
ax.bar(x + w, _slog(tab["empirical_calendar_per_unit"]), w, label="empirique calendrier (60 min)")
ax.axhline(0, color="grey", lw=.8); ax.set_xticks(x)
ax.set_xticklabels(tab["sensor"], rotation=25, ha="right", fontsize=8)
ax.set_ylabel("sensibilite  sign(.)*log10|.|")
ax.set_title("Sensibilite SO2 par levier : 3 mesures independantes"); ax.legend(fontsize=8)
fig.tight_layout(); show_fig(fig, FIG / "phase2_sensibilite.png")

# %% [markdown]
# ## 2. Test de bimodalite (sur chaque candidat, donnees propres)

# %%
bim_rows = []
for s in CFG["phase2"]["lever_candidates"]:
    b = S.bimodality_test(df_blk[s], CFG)
    bim_rows.append({"sensor": s, "is_bimodal": b.get("is_bimodal"),
                     "bic_gain": b.get("bic_gain"), "separation_sigmas": b.get("separation_sigmas"),
                     "low_mode_fraction": b.get("low_mode_fraction"),
                     "component_means": b.get("component_means_phys")})
    print(f"  {s:20s} bimodal={b.get('is_bimodal')}  "
          f"bic_gain={b.get('bic_gain', float('nan')):.1f}  sep={b.get('separation_sigmas', float('nan')):.2f}  "
          f"part_mode_bas={b.get('low_mode_fraction', float('nan')):.3f}  "
          f"modes={[round(v,2) for v in b.get('component_means_phys', [])]}")
bim = pd.DataFrame(bim_rows)
bim.to_csv(OUT / "phase2_bimodalite.csv", index=False)
CK.append(S.checkpoint("Capteurs bimodaux parmi les candidats",
                       int(bim["is_bimodal"].fillna(False).sum())))

fig, axes = plt.subplots(1, len(CFG["phase2"]["lever_candidates"]), figsize=(4*len(bim), 3))
for ax, s in zip(np.atleast_1d(axes), CFG["phase2"]["lever_candidates"]):
    ax.hist(pd.to_numeric(df_blk[s], errors="coerce").dropna(), bins=80, color="#4573b3")
    flag = bool(bim.loc[bim.sensor == s, "is_bimodal"].iloc[0])
    ax.set_title(f"{s}\n{'BIMODAL' if flag else 'unimodal'}", fontsize=8)
    ax.tick_params(labelsize=7)
fig.tight_layout(); show_fig(fig, FIG / "phase2_distributions.png")

# %% [markdown]
# ## 3. Leviers effectivement utilisables par le controleur
#
# Condition : accord de signe ET amplitude coherente ET distribution non bimodale.

# %%
bimap = dict(zip(bim["sensor"], bim["is_bimodal"].fillna(False)))
usable = []
for _, r in tab.iterrows():
    s = r["sensor"]
    ok = bool(r["retained_for_controller"]) and not bool(bimap.get(s, False))
    if ok:
        usable.append(s)
    print(f"  {s:20s} accord_signe={r['sign_agree']}  amplitude_ok={r['amplitude_ok']}  "
          f"bimodal={bimap.get(s)}  -> {'UTILISABLE' if ok else 'ecarte'}")
print(f"\n-> leviers utilisables par le controleur : {usable if usable else 'AUCUN'}")
CK.append(S.checkpoint("Leviers utilisables (signe+amplitude+non bimodal)", len(usable)))

# %% [markdown]
# ## 4. Bornes / reglage habituel / vitesse max -- mesures sur les donnees

# %%
limits = {}
for s in set(usable) | set(CFG["phase2"]["lever_candidates"]):
    lim = S.measure_actuator_limits(df_blk, s, CFG)
    limits[s] = lim
    print(f"  {s:20s} bornes=[{lim['bound_low']:.3f}, {lim['bound_high']:.3f}]  "
          f"habituel={lim['u_habituel']:.3f}  vitesse_max/min={lim['slew_max']:.4f}  std={lim['std']:.3f}")
S.save_json(limits, OUT / "phase2_bornes_actionneurs.json")

# %% [markdown]
# ## 5. Echelle de seuils SO2 (le seuil reglementaire reel n'est pas connu)

# %%
so2_clean = pd.to_numeric(df_blk[CFG["data"]["target_col"]], errors="coerce").dropna()
thr_pcts = CFG["phase2"]["controller"]["threshold_percentiles"]
thresholds = {p: float(np.percentile(so2_clean, p)) for p in thr_pcts}
print("  seuils candidats (centiles de la distribution SO2 observee) :")
for p, v in thresholds.items():
    print(f"    P{p:<3d} = {v:.1f} ppm   (part du temps au-dessus : "
          f"{100*(so2_clean > v).mean():.1f} %)")

# %% [markdown]
# ## 6. Controleur MPC simule
#
# Periode de simulation = dernier segment temporel (20 % les plus recents des
# fenetres valides). Reference "sans commande" = SO2(t+h) reellement observe.

# %%
n = len(W_H); cut = int(n * 0.80)
W_sim = W_H.iloc[cut:].reset_index(drop=True)
print(f"periode de simulation : {W_sim[CFG['data']['date_col']].min()} -> "
      f"{W_sim[CFG['data']['date_col']].max()}  ({len(W_sim)} pas de decision)")


def levers_payload(sensor_list):
    out = []
    for s in sensor_list:
        e = float(tab.loc[tab.sensor == s, "empirical_sustained_per_unit"].iloc[0])
        lim = limits[s]
        out.append({"sensor": s, "gain_per_unit": e,
                    "lo": lim["bound_low"], "hi": lim["bound_high"],
                    "slew": lim["slew_max"], "u_habituel": lim["u_habituel"]})
    return out


scenarios = {}
if usable:
    scenarios["valides_seuls"] = usable
all_cands = CFG["phase2"]["lever_candidates"]
if set(all_cands) != set(usable):
    scenarios["tous_candidats"] = all_cands

sim_results = []
for scen_name, sensor_list in scenarios.items():
    lv = levers_payload(sensor_list)
    for p, thr in thresholds.items():
        r = S.simulate_controller(W_sim, lv, CFG, thr)
        r["scenario"] = scen_name
        r["threshold_pct"] = p
        sim_results.append(r)
        base, ctrl = r["baseline_realized"], r["controlled"]
        print(f"  [{scen_name:15s}] P{p:<3d} (seuil {thr:7.1f}) : "
              f"depassement cumule {base['sum_exceed']:.0f} -> {ctrl['sum_exceed']:.0f} "
              f"({r['sum_exceed_reduction_pct']:+.1f} %)   "
              f"temps>seuil {base['n_above']} -> {ctrl['n_above']}   "
              f"effort(sum u^2)={r['effort_sum_u2']:.1f}   "
              f"saturation leviers={[round(v,2) for v in r['u_saturation_frac']]}")
S.save_json(sim_results, OUT / "phase2_simulation_controleur.json")

# %% [markdown]
# ## 7. Transparence : l'optimiseur sature-t-il les leviers non valides ?

# %%
if "tous_candidats" in scenarios:
    nonval = [s for s in all_cands if s not in usable]
    print(f"leviers non valides inclus dans 'tous_candidats' : {nonval}")
    for r in sim_results:
        if r["scenario"] != "tous_candidats":
            continue
        sat = dict(zip(all_cands, r["u_saturation_frac"]))
        flag = any(sat[s] > 0.5 for s in nonval)
        print(f"  P{r['threshold_pct']:<3d} : saturation des non-valides "
              f"{{{', '.join(f'{s}:{sat[s]:.2f}' for s in nonval)}}}  "
              f"{'<-- gain porte par des signaux non fiables' if flag else ''}")
else:
    print("tous les candidats sont utilisables -- pas de scenario 'tous_candidats' distinct.")

# %% [markdown]
# ## 8. Figure : trajectoire avec / sans commande (meilleur scenario, seuil P95)

# %%
pick = [r for r in sim_results if r["threshold_pct"] == 95]
if pick:
    r = pick[0]
    thr = thresholds[95]
    real_future = W_sim[CFG["data"]["target_col"]].to_numpy() + W_sim["y"].to_numpy()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(real_future, lw=.6, label="SO2(t+h) reel (sans commande)")
    ax.axhline(thr, color="red", ls="--", lw=1, label=f"seuil P95 = {thr:.0f}")
    ax.set_title(f"Reference realisee vs seuil -- scenario '{r['scenario']}'")
    ax.set_xlabel("pas de decision"); ax.set_ylabel("SO2 (ppm)"); ax.legend(fontsize=8)
    fig.tight_layout(); show_fig(fig, FIG / "phase2_trajectoire_P95.png")

# %%
verdict = {
    "modele": "regression lineaire Lasso (enrichie, h=55)",
    "sensibilite_leviers": tab.to_dict(orient="records"),
    "bimodalite": bim.to_dict(orient="records"),
    "leviers_utilisables": usable,
    "seuils": thresholds,
    "n_pas_simulation": len(W_sim),
    "resultats_controleur": sim_results,
}
S.save_json(verdict, OUT / "phase2_verdict.json")
S.save_json(CK, OUT / "checkpoints_phase2.json")

print("\n================= TABLEAU DE CONTROLE -- PHASE 2 =================")
for c in CK:
    print(f"  [{c['status']:5s}] {c['name']:46s} = {c['value']}")
print("================================================================")
