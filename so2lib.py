"""
so2lib.py -- Fonctions du pipeline de prevision des emissions SO2.
Toute la logique est ici ; les scripts / notebooks ne font qu'orchestrer et
afficher. Aucun parametre en dur -> config.yaml.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# =============================================================================
# 0. Config / utilitaires
# =============================================================================

_REFERENCE_CHECKS = True


def load_config(path: str | Path = "config.yaml") -> dict:
    global _REFERENCE_CHECKS
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    _REFERENCE_CHECKS = bool(cfg.get("reference_checks", True))
    return cfg


def checkpoint(name: str, got, expected=None, tol=None, pct=False) -> dict:
    """Affiche 'valeur calculee vs repere' et renvoie un dict pour le JSON.
    Les reperes (valeurs des donnees reelles) sont ignores si reference_checks=false."""
    if not _REFERENCE_CHECKS:
        expected, tol = None, None
    suffix = " %" if pct else ""
    line = f"  [CHECK] {name:42s} = {got}{suffix}"
    status = "info"
    if expected is not None:
        line += f"   (repere doc : {expected}{suffix})"
        if tol is not None:
            try:
                ok = abs(float(got) - float(expected)) <= float(tol)
                status = "ok" if ok else "ECART"
                if not ok:
                    line += "   [ECART -- COMPRENDRE AVANT DE CONTINUER]"
            except (TypeError, ValueError):
                pass
    print(line)
    return {"name": name, "value": got, "expected": expected, "status": status}


def save_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)


# =============================================================================
# 1. Chargement + audit
# =============================================================================

def load_raw(cfg: dict) -> pd.DataFrame:
    d = cfg["data"]
    df = pd.read_csv(d["csv_path"])
    df[d["date_col"]] = pd.to_datetime(df[d["date_col"]])
    df = df.sort_values(d["date_col"]).reset_index(drop=True)
    return df


def audit(df: pd.DataFrame, cfg: dict) -> dict:
    d = cfg["data"]
    date_col = d["date_col"]
    rep: dict = {}
    rep["shape"] = list(df.shape)
    rep["date_min"] = str(df[date_col].min())
    rep["date_max"] = str(df[date_col].max())
    rep["dup_timestamps"] = int(df[date_col].duplicated().sum())

    step = df[date_col].diff().dropna()
    rep["step_counts"] = {str(k): int(v) for k, v in step.value_counts().head(5).items()}
    rep["raw_gaps_gt_1min"] = int((step != pd.Timedelta(minutes=1)).sum())

    rep["nan_per_col"] = {c: int(df[c].isna().sum()) for c in df.columns}

    cols = [c for c in df.columns if c != date_col]
    rep["col_stats"] = {}
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        vc = s.value_counts()
        top_v, top_n = (float(vc.index[0]), int(vc.iloc[0])) if len(vc) else (None, 0)
        rep["col_stats"][c] = {
            "min": float(s.min()), "max": float(s.max()),
            "n_unique": int(s.nunique()),
            "top_value": top_v, "top_count": top_n,
            "n_negative": int((s < 0).sum()),
        }
    return rep


# =============================================================================
# 2. Nettoyage -- codes defaut deguises en mesures
# =============================================================================

def discover_fault_codes(df: pd.DataFrame, cfg: dict) -> dict:
    """
    Une valeur est un code defaut si elle est repetee >= min_repeats fois ET
    viole une borne physique du capteur. Renvoie {col: [(valeur, count, pct), ...]}.
    """
    cl = cfg["cleaning"]
    min_rep = cl["sentinel_discovery"]["min_repeats"]
    bounds = cl["physical_bounds"]
    n = len(df)
    found: dict = {}
    for c in cfg["data"]["raw_channels"]:
        s = pd.to_numeric(df[c], errors="coerce")
        lo = bounds.get(c, {}).get("min", -np.inf)
        hits = []
        vc = s.value_counts()
        for v, cnt in vc.items():
            if cnt >= min_rep and (v < lo):
                hits.append((float(v), int(cnt), round(100 * cnt / n, 3)))
        if hits:
            found[c] = sorted(hits, key=lambda t: -t[1])
    return found


def apply_cleaning(df: pd.DataFrame, cfg: dict):
    """
    Sentinelles (decouvertes + connues) + bornes physiques -> NaN, PUIS dropna().
    Aucune imputation / interpolation / clip. Renvoie (df_propre, rapport).
    """
    cl = cfg["cleaning"]
    d = cfg["data"]
    rtol, atol = cl["isclose_rtol"], cl["isclose_atol"]
    channels = d["raw_channels"]
    df = df.copy()
    for c in channels:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    n0 = len(df)
    rep: dict = {"rows_in": n0}

    # -- doublons d'horodatage
    if cl.get("drop_duplicate_timestamps", True):
        dups = int(df[d["date_col"]].duplicated(keep="first").sum())
        df = df.drop_duplicates(subset=[d["date_col"]], keep="first").reset_index(drop=True)
        rep["dup_dropped"] = dups
    n1 = len(df)

    # -- taux de suppression avec dropna() SEUL (reference)
    rep["drop_rate_dropna_only_pct"] = round(100 * (1 - df.dropna().shape[0] / n1), 3)

    # -- codes defaut -> NaN (decouverte + liste connue), np.isclose
    discovered = discover_fault_codes(df, cfg)
    known = cl.get("known_sentinels", [])
    nan_added = {}
    for c in channels:
        col = df[c].to_numpy(dtype=float)
        mask = np.zeros(len(col), dtype=bool)
        cand = [v for (v, _, _) in discovered.get(c, [])] + list(known)
        for v in cand:
            with np.errstate(invalid="ignore"):
                mask |= np.isclose(col, v, rtol=rtol, atol=atol, equal_nan=False)
        if mask.any():
            df.loc[mask, c] = np.nan
            nan_added[c] = int(mask.sum())
    rep["sentinel_nan_added"] = nan_added
    rep["discovered_sentinels"] = discovered

    # -- bornes physiques -> NaN
    bound_nan = {}
    for c, b in cl["physical_bounds"].items():
        if c not in df.columns:
            continue
        m = pd.Series(False, index=df.index)
        if "min" in b:
            m |= df[c] < b["min"]
        if "max" in b:
            m |= df[c] > b["max"]
        if m.any():
            df.loc[m, c] = np.nan
            bound_nan[c] = int(m.sum())
    rep["bound_nan_added"] = bound_nan

    # -- regle unique : toute ligne avec un NaN dans une colonne -> supprimee
    keep_cols = [d["date_col"]] + channels
    df_clean = df[keep_cols].dropna().reset_index(drop=True)
    rep["rows_out"] = len(df_clean)
    rep["drop_rate_total_pct"] = round(100 * (1 - len(df_clean) / n1), 3)
    rep["kept_pct"] = round(100 * len(df_clean) / n1, 3)
    return df_clean, rep


# =============================================================================
# 3. Blocs temporels continus
# =============================================================================

def make_blocks(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    date_col = cfg["data"]["date_col"]
    freq = pd.Timedelta(minutes=cfg["blocks"]["freq_minutes"])
    df = df.sort_values(date_col).reset_index(drop=True).copy()
    df["block_id"] = (df[date_col].diff() != freq).cumsum()
    return df


def block_stats(df: pd.DataFrame, cfg: dict) -> dict:
    date_col = cfg["data"]["date_col"]
    g = df.groupby("block_id")
    lens = g.size()
    return {
        "n_blocks": int(df["block_id"].nunique()),
        "block_len_min": int(lens.min()),
        "block_len_median": float(lens.median()),
        "block_len_max": int(lens.max()),
        "block_len_mean": round(float(lens.mean()), 1),
        "coverage_days": round((df[date_col].max() - df[date_col].min()).total_seconds() / 86400, 2),
        "rows": int(len(df)),
    }


# =============================================================================
# 4. Grille de faisabilite -- choisir la fenetre de lookback
# =============================================================================

def feasibility_grid(df_blocks: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Pour chaque (lookback L, horizon H) : nb de fenetres exploitables (L min
    d'historique continu ET H min de futur continu dans le meme bloc), + part
    des fenetres concentree sur les N derniers jours.
    """
    date_col = cfg["data"]["date_col"]
    fg = cfg["feasibility_grid"]
    recent_days = fg["recent_days_window"]
    t_end = df_blocks[date_col].max()
    recent_cut = t_end - pd.Timedelta(days=recent_days)

    blocks = [(bid, sub) for bid, sub in df_blocks.groupby("block_id")]
    rows = []
    for L in fg["lookback_candidates_min"]:
        for H in fg["horizon_candidates_min"]:
            n_win, n_recent, first_dt, last_dt = 0, 0, None, None
            for _, sub in blocks:
                m = len(sub)
                k = max(0, m - L - H)          # positions L .. m-1-H
                if k == 0:
                    continue
                n_win += k
                wdates = sub[date_col].iloc[L:m - H]
                n_recent += int((wdates >= recent_cut).sum())
                first_dt = wdates.iloc[0] if first_dt is None else min(first_dt, wdates.iloc[0])
                last_dt = wdates.iloc[-1] if last_dt is None else max(last_dt, wdates.iloc[-1])
            rows.append({
                "lookback_min": L, "horizon_min": H,
                "n_windows": n_win,
                "pct_in_last_%dd" % recent_days: round(100 * n_recent / n_win, 1) if n_win else np.nan,
                "span_days": round((last_dt - first_dt).total_seconds() / 86400, 1) if n_win else np.nan,
            })
    return pd.DataFrame(rows)


# =============================================================================
# 5. Features -- Delta_T + lags + rolling, PAR BLOC
# =============================================================================

def _feat_lists(cfg: dict):
    """Retourne (lags, rolls, rstd, diffs, dt_lags, time_feats) filtres par lookback."""
    f = cfg["features"]
    L = f["lookback_min"]
    lags = [k for k in f["lags_min"] if k <= L]
    rolls = [w for w in f["rolling_min"] if w <= L]
    rstd = [w for w in f.get("rolling_std_min", []) if w <= L]
    diffs = [k for k in f.get("diff_min", []) if k <= L]
    dt_lags = [k for k in f.get("delta_t_lags_min", []) if k <= L]
    time_feats = bool(f.get("add_time_features", False))
    return lags, rolls, rstd, diffs, dt_lags, time_feats


def feature_columns(cfg: dict) -> list[str]:
    f = cfg["features"]
    d = cfg["data"]
    tgt = d["target_col"]
    chans = d["raw_channels"]
    lags, rolls, rstd, diffs, dt_lags, time_feats = _feat_lists(cfg)
    dt_pairs = f["delta_t_pairs"]
    cols: list[str] = []
    for c in chans:                                   # valeurs courantes
        if c == tgt and not f["include_raw_target_as_feature"]:
            continue
        cols.append(c)
    for i in dt_pairs:                                # Delta_T courant
        cols.append(f"Delta_T_{i}")
    for c in chans:                                   # lags voies brutes
        for k in lags:
            cols.append(f"{c}_lag{k}")
    for c in chans:                                   # moyennes mobiles
        for w in rolls:
            cols.append(f"{c}_roll{w}")
    for c in chans:                                   # ecarts-types mobiles (volatilite)
        for w in rstd:
            cols.append(f"{c}_rstd{w}")
    for c in chans:                                   # variations (momentum)
        for k in diffs:
            cols.append(f"{c}_diff{k}")
    for i in dt_pairs:                                # lags de Delta_T
        for k in dt_lags:
            cols.append(f"Delta_T_{i}_lag{k}")
    if time_feats:                                    # saisonnalite (diurne / hebdo)
        cols += ["tod_sin", "tod_cos", "dow_sin", "dow_cos"]
    return cols


def build_features(df_blocks: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    f = cfg["features"]
    d = cfg["data"]
    date_col = d["date_col"]
    chans = d["raw_channels"]
    lags, rolls, rstd, diffs, dt_lags, time_feats = _feat_lists(cfg)

    out = df_blocks.copy()
    g = out.groupby("block_id", sort=False)

    for i in f["delta_t_pairs"]:
        out[f"Delta_T_{i}"] = out[f"OUTLET_TEMP_MASS_{i}"] - out[f"INLET_TEMP_MASS_{i}"]

    new = {}
    for c in chans:
        gc = g[c]
        for k in lags:
            new[f"{c}_lag{k}"] = gc.shift(k)
        for w in rolls:
            new[f"{c}_roll{w}"] = gc.transform(lambda s, w=w: s.rolling(w, min_periods=w).mean())
        for w in rstd:
            new[f"{c}_rstd{w}"] = gc.transform(lambda s, w=w: s.rolling(w, min_periods=w).std())
        for k in diffs:
            new[f"{c}_diff{k}"] = out[c] - gc.shift(k)
    if dt_lags:
        gd = out.groupby("block_id", sort=False)
        for i in f["delta_t_pairs"]:
            for k in dt_lags:
                new[f"Delta_T_{i}_lag{k}"] = gd[f"Delta_T_{i}"].shift(k)
    if time_feats:
        minute = out[date_col].dt.hour * 60 + out[date_col].dt.minute
        new["tod_sin"] = np.sin(2 * np.pi * minute / 1440.0)
        new["tod_cos"] = np.cos(2 * np.pi * minute / 1440.0)
        dow = out[date_col].dt.dayofweek + minute / 1440.0
        new["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
        new["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    out = pd.concat([out, pd.DataFrame(new, index=out.index)], axis=1)
    return out


def build_target(df_feat: pd.DataFrame, horizon: int, cfg: dict) -> pd.Series:
    """y_h = SO2[t+h] - SO2[t], calcule PAR BLOC (shift a l'interieur du bloc)."""
    tgt = cfg["data"]["target_col"]
    fut = df_feat.groupby("block_id", sort=False)[tgt].shift(-horizon)
    return fut - df_feat[tgt]


def valid_windows(df_feat: pd.DataFrame, y: pd.Series, cfg: dict) -> pd.DataFrame:
    """Lignes ou TOUTES les features et la cible existent (lookback + horizon complets)."""
    cols = feature_columns(cfg)
    date_col = cfg["data"]["date_col"]
    tgt = cfg["data"]["target_col"]
    m = df_feat[cols].notna().all(axis=1) & y.notna()
    W = df_feat.loc[m, [date_col, "block_id", tgt] + cols].copy()
    W["y"] = y.loc[m].to_numpy()
    return W.sort_values(date_col).reset_index(drop=True)


# =============================================================================
# 6. Metriques + reference naive
# =============================================================================

def metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"mae": mae, "rmse": rmse, "r2": r2}


def naive_metrics(y_true) -> dict:
    """Persistance : delta predit = 0."""
    return metrics(y_true, np.zeros_like(np.asarray(y_true, float)))


# =============================================================================
# 7. Validation glissante (walk-forward) + embargo
# =============================================================================

def _embargo_min(cfg: dict) -> int:
    e = cfg["validation"]["embargo_min"]
    if e == "auto":
        return int(cfg["features"]["lookback_min"] + cfg["target"]["horizon_min"])
    return int(e)


def walk_forward_folds(W: pd.DataFrame, cfg: dict, embargo_min: int | None = None):
    """
    Replis 'expanding' : train = tout le passe (moins l'embargo), test = segment
    suivant. Plancher min_train_frac avant le 1er repli. Embargo par horodatage.
    Rend des tuples (k, train_pos, test_pos).
    """
    v = cfg["validation"]
    date_col = cfg["data"]["date_col"]
    n = len(W)
    emb = pd.Timedelta(minutes=embargo_min if embargo_min is not None else _embargo_min(cfg))
    min_train = int(np.floor(v["min_train_frac"] * n))
    n_folds = v["n_folds"]
    rest = n - min_train
    fold = rest // n_folds
    dates = W[date_col].to_numpy()
    for k in range(n_folds):
        test_start = min_train + k * fold
        test_end = n if k == n_folds - 1 else min_train + (k + 1) * fold
        if test_end - test_start <= 0:
            continue
        t0 = dates[test_start]
        cutoff = t0 - emb
        train_pos = np.where(W[date_col].to_numpy() <= cutoff)[0]
        train_pos = train_pos[train_pos < test_start]
        test_pos = np.arange(test_start, test_end)
        if len(train_pos) < 50:
            continue
        yield k, train_pos, test_pos


def _fit_predict(Xtr, ytr, Xte, kind, cfg):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import (LinearRegression, LassoCV, RidgeCV,
                                      ElasticNetCV)
    from sklearn.model_selection import TimeSeriesSplit

    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    info = {}
    cv = TimeSeriesSplit(n_splits=cfg["validation"]["ts_split_n"])
    alphas = 10.0 ** np.array(cfg["validation"]["alpha_grid_log10"], dtype=float)

    if kind == "linear":
        mdl = LinearRegression()
    elif kind == "ridge":
        mdl = RidgeCV(alphas=np.logspace(-3, 4, 30))
    elif kind == "lasso":
        mdl = LassoCV(alphas=alphas, cv=cv, max_iter=50000, n_jobs=-1)
    elif kind == "lasso_fixed":
        from sklearn.linear_model import Lasso
        mdl = Lasso(alpha=float(cfg["validation"].get("fixed_alpha", 1.0)), max_iter=50000)
    elif kind == "elasticnet":
        mdl = ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8, 0.95],
                           alphas=alphas, cv=cv, max_iter=20000, n_jobs=-1)
    elif kind == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor
        mdl = HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=600, max_leaf_nodes=31,
            min_samples_leaf=40, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.15,
            random_state=cfg["random_seed"])
    elif kind == "xgb":
        from xgboost import XGBRegressor
        mdl = XGBRegressor(n_estimators=700, max_depth=5, learning_rate=0.03,
                           subsample=0.8, colsample_bytree=0.8,
                           reg_lambda=2.0, n_jobs=-1,
                           random_state=cfg["random_seed"])
    else:
        raise ValueError(kind)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mdl.fit(Xtr_s, ytr)
    if hasattr(mdl, "alpha_"):
        info["alpha"] = float(mdl.alpha_)
    if hasattr(mdl, "l1_ratio_"):
        info["l1_ratio"] = float(mdl.l1_ratio_)
    if hasattr(mdl, "coef_"):
        info["n_nonzero"] = int(np.sum(np.abs(np.ravel(mdl.coef_)) > 1e-12))
    return mdl.predict(Xte_s), info, (mdl, sc)


def run_walk_forward(W: pd.DataFrame, cfg: dict, kind: str = "linear",
                     embargo_min: int | None = None, verbose: bool = False) -> pd.DataFrame:
    cols = feature_columns(cfg)
    X = W[cols].to_numpy(float)
    y = W["y"].to_numpy(float)
    rows = []
    for k, tr, te in walk_forward_folds(W, cfg, embargo_min):
        yp, info, _ = _fit_predict(X[tr], y[tr], X[te], kind, cfg)
        mm, nm = metrics(y[te], yp), naive_metrics(y[te])
        rows.append({
            "fold": k, "n_train": len(tr), "n_test": len(te),
            "r2": mm["r2"], "mae": mm["mae"], "rmse": mm["rmse"],
            "naive_r2": nm["r2"], "naive_mae": nm["mae"], "naive_rmse": nm["rmse"],
            "beats_naive_all3": (mm["mae"] < nm["mae"] and mm["rmse"] < nm["rmse"] and mm["r2"] > nm["r2"]),
            **info,
        })
        if verbose:
            print(f"    fold {k}: n_train={len(tr):6d} n_test={len(te):5d} "
                  f"R2={mm['r2']:.4f} (naive {nm['r2']:.4f})  MAE={mm['mae']:.2f} (naive {nm['mae']:.2f})")
    return pd.DataFrame(rows)


def walk_forward_pooled(W: pd.DataFrame, cfg: dict, kind: str = "linear",
                        embargo_min: int | None = None):
    """Comme run_walk_forward mais renvoie aussi le R2/MAE POOLE (tous les points
    test concatenes) -- moins sensible a un petit repli faible que la moyenne des R2."""
    cols = feature_columns(cfg)
    X = W[cols].to_numpy(float)
    y = W["y"].to_numpy(float)
    rows, yt_all, yp_all = [], [], []
    for k, tr, te in walk_forward_folds(W, cfg, embargo_min):
        yp, info, _ = _fit_predict(X[tr], y[tr], X[te], kind, cfg)
        mm, nm = metrics(y[te], yp), naive_metrics(y[te])
        rows.append({"fold": k, "n_train": len(tr), "n_test": len(te),
                     "r2": mm["r2"], "mae": mm["mae"], "rmse": mm["rmse"],
                     "naive_mae": nm["mae"], "naive_rmse": nm["rmse"],
                     "beats_naive_all3": (mm["mae"] < nm["mae"] and mm["rmse"] < nm["rmse"]
                                          and mm["r2"] > nm["r2"]), **info})
        yt_all.append(y[te]); yp_all.append(yp)
    df_folds = pd.DataFrame(rows)
    yt_all = np.concatenate(yt_all); yp_all = np.concatenate(yp_all)
    pooled = metrics(yt_all, yp_all)
    pooled_naive = naive_metrics(yt_all)
    return df_folds, {"pooled_r2": pooled["r2"], "pooled_mae": pooled["mae"],
                      "pooled_rmse": pooled["rmse"],
                      "pooled_naive_mae": pooled_naive["mae"],
                      "pooled_mae_gain_pct": 100 * (1 - pooled["mae"] / pooled_naive["mae"])}


def summarize_folds(df_folds: pd.DataFrame) -> dict:
    if df_folds.empty:
        return {}
    return {
        "n_folds": int(len(df_folds)),
        "r2_mean": float(df_folds["r2"].mean()), "r2_std": float(df_folds["r2"].std(ddof=0)),
        "mae_mean": float(df_folds["mae"].mean()), "naive_mae_mean": float(df_folds["naive_mae"].mean()),
        "rmse_mean": float(df_folds["rmse"].mean()), "naive_rmse_mean": float(df_folds["naive_rmse"].mean()),
        "mae_gain_pct": float(100 * (1 - df_folds["mae"].mean() / df_folds["naive_mae"].mean())),
        "beats_naive_all_folds": bool(df_folds["beats_naive_all3"].all()),
    }


# =============================================================================
# 8. Balayage d'horizon
# =============================================================================

def horizon_sweep(df_feat: pd.DataFrame, cfg: dict, kind: str = "linear") -> pd.DataFrame:
    rows = []
    for h in cfg["target"]["horizon_sweep_min"]:
        y = build_target(df_feat, h, cfg)
        W = valid_windows(df_feat, y, cfg)
        emb = int(cfg["features"]["lookback_min"] + h)
        folds = run_walk_forward(W, cfg, kind=kind, embargo_min=emb)
        s = summarize_folds(folds)
        rows.append({"horizon_min": h, "n_windows": len(W), **s})
        print(f"  h={h:3d} min : n_win={len(W):6d}  R2={s.get('r2_mean', np.nan):.4f} "
              f"+/- {s.get('r2_std', np.nan):.3f}   MAE {s.get('mae_mean', np.nan):.2f} "
              f"vs naive {s.get('naive_mae_mean', np.nan):.2f}")
    return pd.DataFrame(rows)


# =============================================================================
# 9. Verification de fuite -- 4 protocoles + controle negatif
# =============================================================================

def leakage_protocols(W: pd.DataFrame, cfg: dict, kind: str = "linear") -> pd.DataFrame:
    from sklearn.utils import shuffle as sk_shuffle
    cols = feature_columns(cfg)
    date_col = cfg["data"]["date_col"]
    X = W[cols].to_numpy(float)
    y = W["y"].to_numpy(float)
    n = len(W)
    frac = cfg["leakage"]["test_frac"]
    seed = cfg["random_seed"]
    emb_rows = _embargo_min(cfg)
    out = []

    # A -- melange aleatoire 80/20
    Xs, ys = sk_shuffle(X, y, random_state=seed)
    cut = int(n * (1 - frac))
    yp, _, _ = _fit_predict(Xs[:cut], ys[:cut], Xs[cut:], kind, cfg)
    out.append({"protocol": "A_shuffle_8020", **metrics(ys[cut:], yp)})

    # B -- chronologique, sans embargo
    cut = int(n * (1 - frac))
    yp, _, _ = _fit_predict(X[:cut], y[:cut], X[cut:], kind, cfg)
    out.append({"protocol": "B_chrono_no_embargo", **metrics(y[cut:], yp)})

    # C -- chronologique + embargo
    t0 = W[date_col].iloc[cut]
    cutoff = t0 - pd.Timedelta(minutes=emb_rows)
    tr = np.where(W[date_col].to_numpy() <= cutoff)[0]
    yp, _, _ = _fit_predict(X[tr], y[tr], X[cut:], kind, cfg)
    out.append({"protocol": "C_chrono_embargo", **metrics(y[cut:], yp)})

    # D -- glissant + embargo (protocole retenu) : moyenne des replis
    folds = run_walk_forward(W, cfg, kind=kind)
    s = summarize_folds(folds)
    out.append({"protocol": "D_walkforward_embargo",
                "mae": s["mae_mean"], "rmse": s["rmse_mean"], "r2": s["r2_mean"]})
    return pd.DataFrame(out)


def negative_control(W: pd.DataFrame, cfg: dict, kind: str = "linear") -> dict:
    """Permuter la cible SEULE. Tout R2 positif = fuite ailleurs dans le pipeline."""
    cols = feature_columns(cfg)
    date_col = cfg["data"]["date_col"]
    X = W[cols].to_numpy(float)
    y = W["y"].to_numpy(float)
    n = len(W)
    frac = cfg["leakage"]["test_frac"]
    cut = int(n * (1 - frac))
    emb = _embargo_min(cfg)
    t0 = W[date_col].iloc[cut]
    tr_all = np.where(W[date_col].to_numpy() <= (t0 - pd.Timedelta(minutes=emb)))[0]
    r2s = []
    for rep in range(cfg["leakage"]["negative_control_repeats"]):
        rng = np.random.default_rng(cfg["random_seed"] + rep)
        yperm = rng.permutation(y)
        yp, _, _ = _fit_predict(X[tr_all], yperm[tr_all], X[cut:], kind, cfg)
        r2s.append(metrics(yperm[cut:], yp)["r2"])
    return {"neg_control_r2_mean": float(np.mean(r2s)),
            "neg_control_r2_all": [float(v) for v in r2s]}


# =============================================================================
# 10. Phase 2 -- sensibilite des leviers
# =============================================================================

def fit_final_model(W: pd.DataFrame, cfg: dict, kind: str = "linear"):
    """
    Modele lineaire (OLS ou Lasso) sur TOUTES les fenetres valides, pour l'analyse
    de leviers. La sensibilite analytique (somme des coefficients) est exacte pour
    les deux -- Lasso donne des coefficients plus stables si features colineaires.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression, LassoCV
    from sklearn.model_selection import TimeSeriesSplit
    cols = feature_columns(cfg)
    X = W[cols].to_numpy(float)
    y = W["y"].to_numpy(float)
    sc = StandardScaler().fit(X)
    if kind == "lasso":
        alphas = 10.0 ** np.array(cfg["validation"]["alpha_grid_log10"], dtype=float)
        cv = TimeSeriesSplit(n_splits=cfg["validation"]["ts_split_n"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mdl = LassoCV(alphas=alphas, cv=cv, max_iter=50000, n_jobs=-1).fit(sc.transform(X), y)
    else:
        mdl = LinearRegression().fit(sc.transform(X), y)
    return mdl, sc, cols


# retro-compatibilite
def fit_final_linear(W: pd.DataFrame, cfg: dict):
    return fit_final_model(W, cfg, kind="linear")


def _sensor_feature_terms(sensor: str, cols: list[str], cfg: dict):
    """
    Renvoie [(col, signe)] : les features qui bougent lors d'un changement SOUTENU
    (permanent) de 'sensor'. => valeur courante + lags + moyennes mobiles (signe +1)
    et Delta_T courant + lags (signe -1 si INLET, +1 si OUTLET).
    Les rolling-std et les diff NE bougent PAS sous un decalage constant -> exclus.
    """
    terms = []
    for c in cols:
        if c == sensor or c.startswith(sensor + "_lag") or c.startswith(sensor + "_roll"):
            terms.append((c, +1.0))
    for i in cfg["features"]["delta_t_pairs"]:
        if sensor == f"INLET_TEMP_MASS_{i}":
            sgn = -1.0
        elif sensor == f"OUTLET_TEMP_MASS_{i}":
            sgn = +1.0
        else:
            continue
        for c in cols:
            if c == f"Delta_T_{i}" or c.startswith(f"Delta_T_{i}_lag"):
                terms.append((c, sgn))
    return terms


def analytic_sensitivity(mdl, scaler, sensor: str, cols: list[str], cfg: dict) -> dict:
    """
    Exact pour un modele lineaire : reponse d'un changement SOUTENU de +1 unite
    physique du capteur = somme des coefficients physiques de ses features.
    coef_physique_j = beta_j (espace standardise) / sigma_j.
    """
    beta = mdl.coef_
    sigma = scaler.scale_
    idx = {c: j for j, c in enumerate(cols)}
    total = 0.0
    contribs = {}
    for c, sign in _sensor_feature_terms(sensor, cols, cfg):
        j = idx[c]
        phys = sign * beta[j] / sigma[j]
        contribs[c] = float(phys)
        total += phys
    return {"sensor": sensor, "analytic_per_unit": float(total), "contribs": contribs,
            "n_terms": len(contribs)}


def empirical_sensitivity(df_feat: pd.DataFrame, W: pd.DataFrame, mdl, scaler,
                          sensor: str, cfg: dict, shift_window_min: int | None = None) -> dict:
    """
    Decale les X dernieres minutes d'historique brut de 'sensor' de delta, recalcule
    TOUTES les features qui en dependent (lags, moyennes mobiles, rolling-std, diffs,
    Delta_T courant + lags), mesure la variation moyenne de la prevision, ramenee a
    +1 unite physique.

    shift_window_min :
      - None  -> valeur du config (phase2.empirical_shift_window_min, p.ex. 60) :
                 test "effet de calendrier" (seuls les retards recents bougent).
      - = lookback_min -> decalage SOUTENU (permanent) : comparable a la mesure
                 analytique. diffs et rolling-std ne bougent alors pas.
    """
    p2 = cfg["phase2"]
    cols = feature_columns(cfg)
    f = cfg["features"]
    L = f["lookback_min"]
    Wm = int(shift_window_min) if shift_window_min is not None else p2["empirical_shift_window_min"]
    lags = [k for k in f["lags_min"] if k <= L]
    rolls = [w for w in f["rolling_min"] if w <= L]
    rstd = [w for w in f.get("rolling_std_min", []) if w <= L]
    diffs = [k for k in f.get("diff_min", []) if k <= L]
    dt_lags = [k for k in f.get("delta_t_lags_min", []) if k <= L]
    idx = {c: j for j, c in enumerate(cols)}
    rng = np.random.default_rng(cfg["random_seed"])

    delta = p2["empirical_shift_sigmas"] * float(
        pd.to_numeric(df_feat[sensor], errors="coerce").std())
    if not np.isfinite(delta) or delta == 0:
        return {"sensor": sensor, "empirical_per_unit": np.nan, "delta_phys": delta}

    df_feat = df_feat.reset_index(drop=True)
    pos_in_block = df_feat.groupby("block_id", sort=False).cumcount().to_numpy()
    sensor_arr = pd.to_numeric(df_feat[sensor], errors="coerce").to_numpy(float)

    date_col = cfg["data"]["date_col"]
    abs_pos = pd.Index(df_feat[date_col]).get_indexer(W[date_col])
    ok = abs_pos >= 0
    abs_pos = abs_pos[ok]
    Wv = W.loc[ok].reset_index(drop=True)

    n_samp = min(p2["empirical_sample_windows"], len(abs_pos))
    sel = rng.choice(len(abs_pos), size=n_samp, replace=False)

    Xsel = Wv[cols].to_numpy(float)[sel]
    Xmod = Xsel.copy()
    base_pred = mdl.predict(scaler.transform(Xsel))

    # couplage Delta_T : sensor -> (indice paire, signe d'effet sur Delta_T)
    dt_pair = None
    for i in cfg["features"]["delta_t_pairs"]:
        if sensor == f"INLET_TEMP_MASS_{i}":
            dt_pair = (i, -1.0)
        elif sensor == f"OUTLET_TEMP_MASS_{i}":
            dt_pair = (i, +1.0)

    has_cur = sensor in idx
    for r, s_i in enumerate(sel):
        t = abs_pos[s_i]
        p = pos_in_block[t]
        seg = sensor_arr[t - p: t + 1].copy()
        j = len(seg) - 1                        # position de t dans le segment
        w0 = max(0, j + 1 - Wm)
        seg[w0:] += delta                       # decalage des Wm dernieres minutes
        # -- toutes les features derivees de 'sensor', recalculees depuis seg modifie
        if has_cur:
            Xmod[r, idx[sensor]] = seg[j]
        for k in lags:
            c = f"{sensor}_lag{k}"
            if c in idx and j - k >= 0:
                Xmod[r, idx[c]] = seg[j - k]
        for w in rolls:
            c = f"{sensor}_roll{w}"
            if c in idx and j - w + 1 >= 0:
                Xmod[r, idx[c]] = seg[j - w + 1: j + 1].mean()
        for w in rstd:
            c = f"{sensor}_rstd{w}"
            if c in idx and j - w + 1 >= 0:
                Xmod[r, idx[c]] = seg[j - w + 1: j + 1].std(ddof=1)
        for k in diffs:
            c = f"{sensor}_diff{k}"
            if c in idx and j - k >= 0:
                Xmod[r, idx[c]] = seg[j] - seg[j - k]
        # -- Delta_T couple : courant + lags (l'apport du capteur bouge si dans la fenetre)
        if dt_pair is not None:
            i, sgn = dt_pair
            dc = f"Delta_T_{i}"
            if dc in idx:
                Xmod[r, idx[dc]] = Xsel[r, idx[dc]] + sgn * delta
            for k in dt_lags:
                c = f"Delta_T_{i}_lag{k}"
                if c in idx:
                    shifted = (j - k) >= w0          # t-k tombe dans la fenetre decalee
                    Xmod[r, idx[c]] = Xsel[r, idx[c]] + (sgn * delta if shifted else 0.0)

    mod_pred = mdl.predict(scaler.transform(Xmod))
    dpred = mod_pred - base_pred
    return {
        "sensor": sensor,
        "delta_phys": float(delta),
        "empirical_at_delta_mean": float(np.mean(dpred)),
        "empirical_per_unit": float(np.mean(dpred) / delta),
        "empirical_per_unit_std": float(np.std(dpred) / delta),
        "n_samples": int(n_samp),
        "shift_window_min": Wm,
    }


def model_uses_sensor(mdl, sensor: str, cols: list[str], cfg: dict) -> bool:
    """True si le modele attribue un coefficient non nul a au moins une feature
    derivee de 'sensor' (valeur courante, lags, rolling, rolling-std, diff, Delta_T)."""
    coef = np.ravel(getattr(mdl, "coef_", np.zeros(len(cols))))
    idx = {c: j for j, c in enumerate(cols)}
    terms = {c for c, _ in _sensor_feature_terms(sensor, cols, cfg)}
    for c in cols:
        if (c == sensor or c.startswith(sensor + "_lag") or c.startswith(sensor + "_roll")
                or c.startswith(sensor + "_rstd") or c.startswith(sensor + "_diff")):
            terms.add(c)
    return any(abs(coef[idx[c]]) > 1e-12 for c in terms if c in idx)


def sign_agreement_table(df_feat, W, mdl, scaler, cfg) -> pd.DataFrame:
    """
    Deux mesures independantes par capteur, + un test de calendrier :
      - analytique          : somme des coefficients (reponse a un changement SOUTENU)
      - empirique_soutenue  : decalage de TOUT le lookback (comparable a l'analytique)
      - empirique_calendrier: decalage des X dernieres minutes seulement (quels retards bougent ?)
    Levier retenu si : le modele utilise le capteur, ET analytique / empirique_soutenue /
    empirique_calendrier s'accordent toutes sur le SIGNE, ET |analytique / empirique_soutenue|
    reste sous amplitude_ratio_warn.
    """
    cols = feature_columns(cfg)
    warn = cfg["phase2"]["amplitude_ratio_warn"]
    L = cfg["features"]["lookback_min"]
    rows = []
    for s in cfg["phase2"]["lever_candidates"]:
        used = model_uses_sensor(mdl, s, cols, cfg)
        a = analytic_sensitivity(mdl, scaler, s, cols, cfg)["analytic_per_unit"]
        e_sus = empirical_sensitivity(df_feat, W, mdl, scaler, s, cfg,
                                      shift_window_min=L)["empirical_per_unit"]
        e_cal = empirical_sensitivity(df_feat, W, mdl, scaler, s, cfg)["empirical_per_unit"]
        signs = {np.sign(v) for v in (a, e_sus, e_cal) if np.isfinite(v) and v != 0}
        n_zero = sum(1 for v in (a, e_sus, e_cal) if (not np.isfinite(v)) or v == 0)
        sign_agree = used and (len(signs) == 1) and (n_zero == 0)
        if np.isfinite(a) and np.isfinite(e_sus) and min(abs(a), abs(e_sus)) > 0:
            ratio = max(abs(a), abs(e_sus)) / min(abs(a), abs(e_sus))
        else:
            ratio = np.nan
        amp_ok = bool(np.isfinite(ratio) and ratio <= warn)
        rows.append({
            "sensor": s,
            "model_uses_sensor": bool(used),
            "analytic_per_unit": a,
            "empirical_sustained_per_unit": e_sus,
            "empirical_calendar_per_unit": e_cal,
            "sign_agree": bool(sign_agree),
            "amplitude_ratio_analytic_vs_sustained": ratio,
            "amplitude_ok": amp_ok,
            "retained_for_controller": bool(sign_agree and amp_ok),
        })
    return pd.DataFrame(rows)


# =============================================================================
# 11. Phase 2 -- bimodalite + limites actionneur
# =============================================================================

def bimodality_test(series: pd.Series, cfg: dict) -> dict:
    from sklearn.mixture import GaussianMixture
    p = cfg["phase2"]["bimodality"]
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy(float)
    if len(x) < 200:
        return {"is_bimodal": False, "reason": "trop peu de points"}
    mu, sd = x.mean(), x.std()
    z = ((x - mu) / (sd if sd else 1.0)).reshape(-1, 1)
    g1 = GaussianMixture(1, random_state=cfg["random_seed"], n_init=3).fit(z)
    g2 = GaussianMixture(2, random_state=cfg["random_seed"], n_init=5).fit(z)
    bic1, bic2 = g1.bic(z), g2.bic(z)
    m = g2.means_.ravel()
    s = np.sqrt(g2.covariances_.ravel())
    wts = g2.weights_.ravel()
    sep = abs(m[0] - m[1]) / max(s.max(), 1e-9)
    is_bi = (bic1 - bic2 > p["bic_gain_min"]
             and sep > p["min_separation_sigmas"]
             and wts.min() > p["min_component_weight"])
    lo_comp = int(np.argmin(m))
    return {
        "is_bimodal": bool(is_bi),
        "bic1": float(bic1), "bic2": float(bic2), "bic_gain": float(bic1 - bic2),
        "separation_sigmas": float(sep),
        "component_means_phys": [float(v) for v in (m * sd + mu)],
        "component_weights": [float(v) for v in wts],
        "low_mode_fraction": float(wts[lo_comp]),
    }


def measure_actuator_limits(df_clean_blocks: pd.DataFrame, sensor: str, cfg: dict) -> dict:
    p = cfg["phase2"]["controller"]
    s = pd.to_numeric(df_clean_blocks[sensor], errors="coerce")
    lo_p, hi_p = p["bound_percentiles"]
    diffs = df_clean_blocks.groupby("block_id", sort=False)[sensor].diff().abs().dropna()
    return {
        "sensor": sensor,
        "bound_low": float(np.percentile(s.dropna(), lo_p)),
        "bound_high": float(np.percentile(s.dropna(), hi_p)),
        "u_habituel": float(s.median()),
        "slew_max": float(np.percentile(diffs, p["slew_percentile"])),
        "std": float(s.std()),
    }


# =============================================================================
# 12. Phase 2 -- controleur MPC simule
# =============================================================================

def simulate_controller(W_test: pd.DataFrame, levers: list[dict], cfg: dict,
                        threshold: float) -> dict:
    """
    Reference 'sans commande' = trajectoire SO2 REELLEMENT observee (colonne cible).
    Le modele ne fournit que l'effet INCREMENTAL valide d'un ajustement de levier :
        SO2_commande(t+h) = SO2_reel(t+h) + sum_S  gain_S * u_adj_S(t)
    Cout par pas = w_over*max(0, pred-seuil)^2 + w_move*dU^2 + w_dev*u_adj^2.
    'levers' : liste de dicts {sensor, gain_per_unit, lo, hi, slew, u_habituel}.
    """
    from scipy.optimize import minimize
    c = cfg["phase2"]["controller"]
    tgt = cfg["data"]["target_col"]
    wov, wmv, wdv = c["w_overshoot"], c["w_move"], c["w_deviation"]

    realized_future = W_test[tgt].to_numpy(float) + W_test["y"].to_numpy(float)  # SO2(t+h) reel
    realized_now = W_test[tgt].to_numpy(float)
    gains = np.array([lv["gain_per_unit"] for lv in levers], float)
    lo = np.array([lv["lo"] - lv["u_habituel"] for lv in levers], float)
    hi = np.array([lv["hi"] - lv["u_habituel"] for lv in levers], float)
    slew = np.array([lv["slew"] for lv in levers], float)

    u_prev = np.zeros(len(levers))
    u_hist, pred_ctrl = [], []
    for t in range(len(W_test)):
        rf = realized_future[t]
        lob = np.maximum(lo, u_prev - slew)
        hib = np.minimum(hi, u_prev + slew)
        lob = np.minimum(lob, hib)

        def cost(u):
            pred = rf + float(gains @ u)
            over = max(0.0, pred - threshold)
            return wov * over ** 2 + wmv * float(np.sum((u - u_prev) ** 2)) + wdv * float(np.sum(u ** 2))

        if np.allclose(lob, hib):
            u_opt = lob.copy()
        else:
            res = minimize(cost, x0=np.clip(u_prev, lob, hib), method="L-BFGS-B",
                           bounds=list(zip(lob, hib)))
            u_opt = res.x
        u_prev = u_opt
        u_hist.append(u_opt.copy())
        pred_ctrl.append(rf + float(gains @ u_opt))

    u_hist = np.array(u_hist)
    pred_ctrl = np.array(pred_ctrl)

    def exc(v):
        e = np.maximum(0.0, v - threshold)
        return {"n_above": int((v > threshold).sum()),
                "sum_exceed": float(e.sum()), "max_exceed": float(e.max() if len(e) else 0.0),
                "mean_level": float(v.mean())}

    base = exc(realized_future)
    ctrl = exc(pred_ctrl)
    return {
        "threshold": float(threshold),
        "n_steps": int(len(W_test)),
        "levers": [lv["sensor"] for lv in levers],
        "baseline_realized": base,
        "controlled": ctrl,
        "sum_exceed_reduction_pct": float(100 * (1 - ctrl["sum_exceed"] / base["sum_exceed"]))
        if base["sum_exceed"] > 0 else np.nan,
        "n_above_reduction_pct": float(100 * (1 - ctrl["n_above"] / base["n_above"]))
        if base["n_above"] > 0 else np.nan,
        "effort_sum_u2": float(np.sum(u_hist ** 2)),
        "effort_mean_abs_u": [float(v) for v in np.mean(np.abs(u_hist), axis=0)],
        "effort_max_abs_u": [float(v) for v in np.max(np.abs(u_hist), axis=0)],
        "u_saturation_frac": [float(np.mean(np.isclose(np.abs(u_hist[:, j]),
                              max(abs(lo[j]), abs(hi[j])), rtol=1e-3)))
                              for j in range(len(levers))],
    }
