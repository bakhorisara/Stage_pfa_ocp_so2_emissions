"""
make_synthetic_data.py -- genere un jeu de donnees SYNTHETIQUE au meme format que
les donnees capteurs industrielles (1 ligne / minute, 12 voies), afin que le
pipeline soit executable sans les donnees reelles (confidentielles).

Ce qui est simule (pour exercer chaque etape du pipeline) :
  * dynamique lente des temperatures + cycle journalier ;
  * SO2 = processus autoregressif tire par OUTLET/INLET_TEMP_MASS_1 et TEMP_FOUR ;
  * TURBO_SPEED bimodal (marche / ralenti) ;
  * trous d'acquisition et codes defaut deguises en mesures (-999.9, ...)
    -> decoupage en blocs continus et nettoyage.

Usage :  python make_synthetic_data.py [--days 14] [--seed 0]
Sortie : data/synthetic_so2.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def ar1(n, phi, sigma, rng):
    x = np.zeros(n)
    e = rng.normal(0, sigma, n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def main(days: int, seed: int, out: Path) -> None:
    rng = np.random.default_rng(seed)
    n = days * 24 * 60
    t = pd.date_range("2025-01-06 00:00", periods=n, freq="min")
    hour = t.hour.to_numpy() + t.minute.to_numpy() / 60
    daily = np.sin(2 * np.pi * hour / 24)

    flow = 20.5 + ar1(n, 0.995, 0.08, rng) + 0.4 * daily
    four = 890 + ar1(n, 0.998, 1.2, rng) + 6 * daily
    turbo_on = (ar1(n, 0.9995, 0.02, rng) > -0.25).astype(float)
    turbo = np.where(turbo_on > 0, 9800 + ar1(n, 0.99, 15, rng), 4200 + ar1(n, 0.99, 25, rng))

    temps = {}
    base_in = {1: 415, 2: 470, 3: 440, 4: 425}
    base_out = {1: 605, 2: 515, 3: 456, 4: 428}
    for i in range(1, 5):
        drift = ar1(n, 0.997, 0.35, rng)
        temps[f"INLET_TEMP_MASS_{i}"] = base_in[i] + drift + 0.02 * (four - 890)
        temps[f"OUTLET_TEMP_MASS_{i}"] = (base_out[i] + 0.8 * drift + ar1(n, 0.99, 0.15, rng)
                                          + 0.01 * (flow - 20.5) * 10)

    # SO2 : AR(1) + effets procede retardes + bruit
    lag = 20
    drv = (6.0 * (np.roll(temps["OUTLET_TEMP_MASS_1"], lag) - 605)
           + 4.0 * (np.roll(temps["INLET_TEMP_MASS_1"], lag) - 415)
           + 1.5 * (np.roll(four, lag) - 890)
           + 25.0 * (np.roll(flow, lag) - 20.5)
           + 60 * daily)
    so2 = np.zeros(n)
    so2[0] = 1640
    noise = rng.normal(0, 35, n)
    for k in range(1, n):
        so2[k] = 1640 + 0.97 * (so2[k - 1] - 1640) + 0.03 * drv[k] * 3 + noise[k]

    df = pd.DataFrame({"Date": t.strftime("%Y-%m-%d %H:%M:%S"),
                       "INPUT_SULF_FLOW": flow, "TEMP_FOUR": four, "TURBO_SPEED": turbo,
                       **{k: temps[k] for k in ["INLET_TEMP_MASS_1", "OUTLET_TEMP_MASS_1",
                                                "INLET_TEMP_MASS_2", "OUTLET_TEMP_MASS_2",
                                                "INLET_TEMP_MASS_3", "OUTLET_TEMP_MASS_3",
                                                "INLET_TEMP_MASS_4", "OUTLET_TEMP_MASS_4"]},
                       "SO2_EMISSIONS": so2})

    # codes defaut deguises en mesures (sequences courtes)
    for _ in range(days * 3):
        c = rng.choice(["SO2_EMISSIONS", "INPUT_SULF_FLOW", "TEMP_FOUR", "OUTLET_TEMP_MASS_2"])
        s = rng.integers(0, n - 60)
        df.loc[s:s + rng.integers(5, 40), c] = -999.9
    # trous d'acquisition (lignes absentes)
    drop = np.zeros(n, bool)
    for _ in range(days * 2):
        s = rng.integers(0, n - 120)
        drop[s:s + rng.integers(2, 90)] = True
    df = df.loc[~drop]

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%.4f")
    print(f"{len(df)} lignes ecrites dans {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("data/synthetic_so2.csv"))
    a = ap.parse_args()
    main(a.days, a.seed, a.out)
