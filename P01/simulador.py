"""Behavioral-bias simulator: main entry point (disposition effect + overconfidence).

Simulator design
----------------
1. Prices (``simlib_precios``): one path of 100 securities over 756 trading days from a
   market factor plus three zero-premium style factors. It is generated before any agent
   exists and every security has zero alpha, so informed trading is impossible by
   construction.
2. Agents (``simlib_agentes``): 1,000 accounts per scenario with starting wealth W_i,
   n_i positions, a baseline sale intensity l_i (liquidity needs), disposition delta_i
   and overprecision kappa_i.
3. Injected mechanism (A, domain-dependent holding hazard), one level below PGR/PLR::

       h0_i = l_i * (1 + 3 * kappa_i) / 252                  daily baseline hazard
       P(sell a position today) = h0_i * (1 + delta_i)     above its average cost
                                = h0_i * (1 - delta_i)     below it (h0_i at a tie)

   Proceeds buy a uniformly random security the same day (no cash drag); every trade
   pays a $10 commission and half of a 20 bp spread. PGR, PLR and turnover are never set:
   they emerge. The confounds keep delta = kappa = 0: equal-weight rebalancing with a
   25 % band (S7) and a false belief in reversal, hazard x5 after a +12 % 40-day run (S8).
4. Estimators (``simlib_estimadores``): Odean PGR/PLR with an account bootstrap, and the
   Barber-Odean turnover regression on gross and net returns (OLS, and 2SLS with the
   ex-ante hazard h0_i as instrument).

Usage
-----
    python simulador.py                # 8 scenarios -> resultados/tabla_simulador.csv
    python simulador.py --n-boot 200   # faster, noisier standard errors

The report (``simulador_sesgos_conductuales.ipynb``) imports this module and adds the
diagnostics, figures and analysis.
"""

# ======================================================================================
# PRE-ANALYSIS (registered 2026-09-10T23:17:59Z, before any estimator existed; never
# edited). Official text: resultados/pre_analisis.md (SHA-256 df22412ecfa7b292...).
# This copy renders its LaTeX as plain characters; the wording is unchanged.
# Role of this file: main simulator. It builds the scenario grid that injects these
#   parameters and runs every estimator whose recovery is predicted below.
# --------------------------------------------------------------------------------------
# [REF] — PRE-ANÁLISIS (registrado antes de ejecutar cualquier estimador; no editado
# después)
#
# Inyectamos, con N = 1,000 cuentas por escenario, 100 valores y 756 días, δ y κ como
# constantes poblacionales (δ ∈ {0, 0.3, 0.8}, κ ∈ {0, 0.3, 0.8}) sobre un riesgo base
# diario h_0=ℓ_i(1+3κ)/252 con ℓ_i ~ U(0.25,1.25) y multiplicadores 1±δ según el dominio
# respecto del costo promedio; los confusores usan δ=κ=0 (rebalanceo: banda de 25 %
# sobre el peso igual, recorte al objetivo y compra de la posición más infraponderada;
# reversión: riesgo × 5 si el rendimiento de 40 días supera 12 %). Esperamos, por la
# aritmética del riesgo condicionado a días de venta, PGR ≈ PLR ≈ E[n]/E[n^2] ≈ 0.05 en
# el nulo y PGR/PLR ≈ (1+δ)/(1-δ) corregido a la baja por la heterogeneidad de
# composición: E1 diferencia con IC que contiene 0 y β_bruta (MCO e IV) con IC que
# contiene 0; E2 diferencia ≈ 0.03 (0.02–0.04), razón 1.5–2.2; E3 diferencia ≈ 0.10
# (0.07–0.13), razón 5–11, y misbehavior esperado: β_bruta^MCO>0 de varios puntos por
# causalidad inversa, suficiente para que β_neta^MCO no sea negativa, mientras la IV
# bruta queda en 0; E4/E5 diferencia en 0 y β_neta ≈ -1.2 pp por unidad de rotación
# anual (rango -0.7 a -1.8) igual en ambos, de modo que anticipamos que el requisito de
# monotonía de |β_neta| de 4 a 5 fallará: la pendiente mide costo por unidad de
# rotación, no la dosis de κ; lo que sí crece es el lastre medio de costos (≈0.8, 1.5 y
# 2.6 %/año en E1, E4, E5); E6 diferencia menor que en E3 (carteras más jóvenes, menos
# acumulación de perdedoras) y razón mayor o igual que en E3, con β_bruta^MCO>0 pero
# diluida respecto de E3; E7 disposición espuria positiva y significativa (razón 2–5
# contando cada recorte como una realización, y al menos 40 % menor con la convención
# fraccional), β_bruta^MCO>0 pequeña; E8 disposición espuria positiva (razón 1.3–3,
# diferencia 0.015–0.05) y β_bruta^MCO>0. En todos los escenarios esperamos
# β_neta-β_bruta=-β_costos exacta con rendimientos aritméticos; marcar a precio de
# ejecución desplaza β_bruta en exactamente -0.20 pp; una ventana de caja de 21 días la
# desplaza ≈ -(21/252) × 2.1 % ≈ -0.2 pp (el mercado realizado de la trayectoria,
# conocido porque los precios se generan antes que los agentes, rindió solo 2.1 %/año),
# probablemente no significativo; la regla de empates afectará menos de 0.1 % de las
# observaciones; FIFO y lote-por-lote atenuarán la diferencia frente al costo promedio
# (la referencia del propio agente); el SE por cuenta será ≈ igual al ingenuo en el nulo
# y mayor cuando δ>0 o hay confusor; la prueba de fuga dará coeficientes nulos (p > 0.05
# a 20 y 120 días) y la diferencia comprado-menos-vendido a 252 días será ≈ 0 frente a
# los −3.2 pp de Odean (1999); en la población heterogénea, corr(δ,κ) con IC que
# contiene 0 y corr(δ, rotación) negativa; con corr(δ,κ)=0.6 la β_bruta^MCO sube
# mientras la IV con κ sigue ≈ 0; la curva de recuperación será creciente y convexa en
# δ, con razón indefinida en δ=1 (PLR=0); y la dispersión de β_bruta entre trayectorias
# de precios independientes superará el SE bootstrap por cuenta (factor 1.2–3) porque
# las cuentas comparten valores.
#
# Antes de este registro solo se ejecutó una prueba mecánica del simulador (identidad
# contable, tiempo de cómputo y rotación media como calibración del DGP); ningún
# estimador de PGR/PLR ni de β estaba implementado. Nada de lo que aparece debajo de
# esta celda se usó para revisarla.
# ======================================================================================

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import simlib_agentes as sa
import simlib_estimadores as se
import simlib_precios as sp

MASTER_SEED = 20260910
MASTER = np.random.SeedSequence(MASTER_SEED)
N_BOOT = 1000
MARKET_CFG, BEH, COSTS = sp.MarketConfig(), sa.BehaviorConfig(), sa.CostConfig()

#: The eight scenarios of the brief. delta and kappa are population constants: every
#: account carries exactly the injected value.
GRID = [sa.ScenarioConfig("S1", "Nulo", 0.0, 0.0),
        sa.ScenarioConfig("S2", "Disposición baja", 0.3, 0.0),
        sa.ScenarioConfig("S3", "Disposición alta", 0.8, 0.0),
        sa.ScenarioConfig("S4", "Rotación baja", 0.0, 0.3),
        sa.ScenarioConfig("S5", "Rotación alta", 0.0, 0.8),
        sa.ScenarioConfig("S6", "Ambos activos", 0.8, 0.8),
        sa.ScenarioConfig("S7", "Confusor: rebalanceo", 0.0, 0.0, "rebalancing"),
        sa.ScenarioConfig("S8", "Confusor: reversión", 0.0, 0.0, "mean_reversion")]
SCN = {s.key: s for s in GRID}
CONFOUND_LABELS = {"none": "—", "rebalancing": "rebalanceo", "mean_reversion": "reversión"}

#: Random-stream map. Every stream is ``child_seed(MASTER, key)``, so each run is
#: reproducible on its own, whatever else ran before it.
STREAMS = [("mercado", "0"), ("escenarios E1–E8 (población y decisiones)", "1–8"),
           ("E9 heterogénea / E9m mal especificada", "9 / 10"),
           ("barrido de δ / barrido de κ", "11–16 / 21–26"),
           ("prueba de invariancia (futuro alterno)", "31"), ("mercados GBM", "40"),
           ("Monte Carlo: población y decisiones / mercado", "500+r / 600+r"),
           ("réplicas de E9 / E9m (misma trayectoria)", "700+r / 720+r"),
           ("bootstrap de cada corrida", "(300, flujo)")]


def build_market(cfg: sp.MarketConfig = MARKET_CFG, stream: int = 0) -> sp.Market:
    """Generate the price universe from its own stream, before any agent exists.

    Args:
        cfg: Market calibration.
        stream: Integer key of the market's random stream (0 = the report's path).

    Returns:
        The realized :class:`simlib_precios.Market`.
    """
    return sp.simulate_market(cfg, np.random.default_rng(sa.child_seed(MASTER, stream)))


def run_scenario(scn: sa.ScenarioConfig, stream: int, market: sp.Market,
                 n_boot: int = N_BOOT, keep_sim: bool = False, quiet: bool = False) -> dict:
    """Simulate one population on ``market`` and apply every estimator to it.

    The population and the daily decisions use sub-streams of ``child_seed(MASTER, stream)``
    and the account bootstrap uses ``child_seed(MASTER, 300, stream)``.

    Args:
        scn: Scenario (injected delta and kappa, confound, cash lag).
        stream: Integer key of the scenario's random stream.
        market: Price universe, generated beforehand.
        n_boot: Account-bootstrap replications.
        keep_sim: Also return the raw ``SimulationResult`` under ``"sim"``.
        quiet: Do not print the timing line.

    Returns:
        The dict of :func:`simlib_estimadores.summarize_run` plus ``"seconds"``.
    """
    seed = sa.child_seed(MASTER, stream)
    start = time.perf_counter()
    sim = sa.simulate_trading(market, sa.draw_population(scn, BEH, seed), scn, BEH, COSTS, seed)
    res = se.summarize_run(sim, market, n_boot,
                           np.random.default_rng(sa.child_seed(MASTER, 300, stream)))
    res["seconds"] = time.perf_counter() - start
    if not quiet:
        print(f"{scn.key:>6s}  {scn.label:<34s} flujo {stream:>4}   {res['seconds']:5.1f} s")
    if keep_sim:
        res["sim"] = sim
    return res


def run_grid(market: sp.Market, n_boot: int = N_BOOT, quiet: bool = False) -> dict[str, dict]:
    """Run the eight scenarios of :data:`GRID`; scenario i uses stream i.

    Args:
        market: Price universe shared by all scenarios (common random numbers).
        n_boot: Account-bootstrap replications per scenario.
        quiet: Do not print timing lines.

    Returns:
        Results of :func:`run_scenario` keyed by scenario key (``"S1"`` ... ``"S8"``).
    """
    return {scn.key: run_scenario(scn, i, market, n_boot, quiet=quiet)
            for i, scn in enumerate(GRID, start=1)}


def slope_row(res: dict, method: str, outcome: str) -> pd.Series:
    """One row of a run's slope table (``method`` "OLS"/"IV", ``outcome`` "gross"/"net"/...)."""
    s = res["slopes"]
    return s[(s["method"] == method) & (s["outcome"] == outcome)].iloc[0]


def results_table(results: dict[str, dict]) -> pd.DataFrame:
    """Injected parameters, recovered estimates and standard errors, one row per scenario.

    PGR, PLR, their difference and ratio use the baseline convention (average cost, a
    partial sale counts as one realization) with 95 % account-bootstrap percentile
    intervals. Slopes are percentage points of annual return per unit of annual
    turnover: OLS with account-bootstrap and HC1 standard errors, and 2SLS with the
    ex-ante hazard as the instrument.

    Args:
        results: Output of :func:`run_grid`, or any dict of :func:`run_scenario` results.

    Returns:
        DataFrame with one row per scenario.
    """
    rows = []
    for key, res in results.items():
        scn, b = res["scenario"], res["conventions"]["avg_count"]
        og, on, ig, inn = (slope_row(res, m, o) for m, o in
                           (("OLS", "gross"), ("OLS", "net"), ("IV", "gross"), ("IV", "net")))
        rows.append({"Esc.": key, "delta": scn.delta, "kappa": scn.kappa,
                     "confusor": CONFOUND_LABELS[scn.confound],
                     "PGR": b["pgr"], "PLR": b["plr"], "dif": b["diff"], "dif_ee": b["diff_se"],
                     "dif_ic_inf": b["diff_lo"], "dif_ic_sup": b["diff_hi"], "razon": b["ratio"],
                     "razon_ee": b["ratio_se"], "razon_ic_inf": b["ratio_lo"],
                     "razon_ic_sup": b["ratio_hi"],
                     "b_bruta": og.slope, "b_bruta_ee_boot": og.se_boot, "b_bruta_ee_hc1": og.se_hc1,
                     "b_neta": on.slope, "b_neta_ee_boot": on.se_boot, "b_neta_ee_hc1": on.se_hc1,
                     "b_iv_bruta": ig.slope, "b_iv_bruta_ee": ig.se_boot, "b_iv_neta": inn.slope,
                     "b_iv_neta_ee": inn.se_boot, "rotacion_media": res["panel"]["turnover"].mean(),
                     "costo_medio_pp": res["panel"]["costs"].mean(), "n_cuentas": len(res["panel"]),
                     "n_obs_odean": b["observations"]})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    """Command line: run the eight scenarios, print and save the results table.

    Args:
        argv: Arguments (default: ``sys.argv``); see ``python simulador.py --help``.
    """
    parser = argparse.ArgumentParser(description="Simulador de sesgos conductuales: 8 escenarios.")
    parser.add_argument("--n-boot", type=int, default=N_BOOT,
                        help=f"réplicas del bootstrap por cuenta (default {N_BOOT})")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parent / "resultados" / "tabla_simulador.csv",
                        help="archivo CSV de salida")
    args = parser.parse_args(argv)
    start = time.perf_counter()
    table = results_table(run_grid(build_market(), args.n_boot))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.round(6).to_csv(args.out, index=False)
    cols = ["Esc.", "delta", "kappa", "confusor", "PGR", "PLR", "dif", "dif_ee", "razon", "razon_ee",
            "b_bruta", "b_bruta_ee_boot", "b_neta", "b_neta_ee_boot", "b_iv_bruta", "b_iv_bruta_ee",
            "rotacion_media"]
    with pd.option_context("display.width", 220, "display.max_columns", 30,
                           "display.float_format", "{:,.4f}".format):
        print("\n" + table[cols].to_string(index=False))
    print(f"\nTabla guardada en {args.out} ({time.perf_counter() - start:.0f} s)")


if __name__ == "__main__":
    main()
