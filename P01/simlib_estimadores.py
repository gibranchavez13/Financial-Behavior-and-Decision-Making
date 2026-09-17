"""Estimators: Odean PGR/PLR accounting and Barber-Odean turnover regressions.

These functions only see what a brokerage export would contain (sale-day
snapshots, trades, daily account values). Ground-truth parameters enter only
where explicitly labelled (the ex-ante instrument and the diagnostics).
"""

# ======================================================================================
# PRE-ANALYSIS (registered 2026-09-10T23:17:59Z, before any estimator existed; never
# edited). Official text: resultados/pre_analisis.md (SHA-256 df22412ecfa7b292...).
# This copy renders its LaTeX as plain characters; the wording is unchanged.
# Role of this file: implements every estimator whose recovery the paragraph predicts
#   (PGR/PLR, account bootstrap, gross/net OLS and IV slopes, leakage test).
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

import numpy as np
import pandas as pd
from scipy import stats

from simlib_agentes import TIE_USD, SimulationResult
from simlib_precios import TRADING_DAYS, Market, forward_returns

COUNT_NAMES = ("G_r", "G_p", "L_r", "L_p")

#: name -> (reference column or "lot", partial-sale rule)
CONVENTIONS: dict[str, tuple[str, str]] = {
    "avg_count": ("avg_cost", "count"),       # baseline (the agent's own reference)
    "avg_fraction": ("avg_cost", "fraction"),
    "fifo_count": ("fifo_cost", "count"),
    "last_count": ("last_cost", "count"),
    "lot_count": ("lot", "count"),
}


def classify(mid: np.ndarray, reference: np.ndarray, tie: float = TIE_USD) -> np.ndarray:
    """Paper gain (+1), paper loss (-1) or tie (0) of a position.

    Args:
        mid: Mid-quote on the sale day.
        reference: Cost basis under the chosen convention.
        tie: Half-width of the tie band in USD (positions inside it are
            excluded from all four counts).

    Returns:
        Integer array of signs.
    """
    gap = mid - reference
    return np.where(gap > tie, 1, np.where(gap < -tie, -1, 0)).astype(np.int8)


def disposition_counts(sim: SimulationResult, convention: str = "avg_count") -> dict:
    """Per-account Odean counts on sale days under one convention.

    Rules shared by every convention: account-days without a sale contribute
    nothing; positions bought on the sale day are not in the snapshot; ties
    are excluded and counted separately. ``count`` records a (partial) sale
    as one realization; ``fraction`` splits the observation into the share
    sold (realized) and the share kept (paper).

    Args:
        sim: Simulation output (uses the ``snapshots`` and ``lots`` tables).
        convention: Key of :data:`CONVENTIONS`.

    Returns:
        Dict with ``counts`` (N, 4) float array ordered as G_r, G_p, L_r, L_p,
        ``ties`` and ``observations`` (totals incl. ties).
    """
    reference, partial = CONVENTIONS[convention]
    tab = sim.tables["lots" if reference == "lot" else "snapshots"]
    ref = tab["cost"] if reference == "lot" else tab[reference]
    sign = classify(tab["mid"], ref)
    held = tab["shares"].astype(float)
    sold = tab["shares_sold"].astype(float)
    realized = (sold > 0).astype(float) if partial == "count" else np.minimum(sold / held, 1.0)
    n = sim.population.n_agents
    acc = tab["account"]
    cols = [(sign == 1) * realized, (sign == 1) * (1.0 - realized),
            (sign == -1) * realized, (sign == -1) * (1.0 - realized)]
    counts = np.column_stack([np.bincount(acc, c, n) for c in cols])
    return {"counts": counts, "ties": int((sign == 0).sum()), "observations": int(sign.size)}


def pgr_plr(total: np.ndarray) -> dict[str, float]:
    """PGR, PLR, their difference and ratio from pooled counts.

    Args:
        total: Array whose last axis is (G_r, G_p, L_r, L_p).

    Returns:
        Dict of floats (or arrays when ``total`` has leading axes).
    """
    total = np.asarray(total, dtype=float)
    pgr = total[..., 0] / (total[..., 0] + total[..., 1])
    plr = total[..., 2] / (total[..., 2] + total[..., 3])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = pgr / plr
    return {"pgr": pgr, "plr": plr, "diff": pgr - plr, "ratio": ratio}


def _summarize_draws(point: dict, draws: dict, alpha: float = 0.05) -> dict[str, float]:
    """Bootstrap SE and percentile CI for each statistic in ``draws``."""
    out = {}
    for k, v in draws.items():
        finite = v[np.isfinite(v)]
        out[k] = float(point[k])
        out[f"{k}_se"] = float(finite.std(ddof=1)) if finite.size > 1 else np.nan
        lo, hi = (np.quantile(finite, [alpha / 2, 1 - alpha / 2])
                  if finite.size else (np.nan, np.nan))
        out[f"{k}_lo"], out[f"{k}_hi"] = float(lo), float(hi)
    return out


def account_weights(n_accounts: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Resampling multiplicities of an account bootstrap.

    Row b says how many times each account appears in replication b. The same
    matrix drives the PGR/PLR bootstrap and the regression bootstrap, so both
    estimators are evaluated on identical resampled populations.

    Args:
        n_accounts: Number of accounts N.
        n_boot: Number of replications B.
        rng: Generator.

    Returns:
        (B, N) integer matrix whose rows sum to N.
    """
    return rng.multinomial(n_accounts, np.full(n_accounts, 1.0 / n_accounts), size=n_boot)


def bootstrap_accounts(counts: np.ndarray, weights: np.ndarray) -> tuple[dict, dict]:
    """Account-level (cluster) bootstrap of PGR, PLR, difference and ratio.

    Accounts are resampled with replacement and the four counts are re-pooled
    inside every replication, so all within-account dependence (the same
    positions counted on many sale days, persistent composition) is kept.

    Args:
        counts: (N, 4) per-account counts.
        weights: (B, N) multiplicities from :func:`account_weights`.

    Returns:
        ``(summary, draws)``: point estimates with SEs and 95 % percentile CIs,
        and the raw replicate arrays.
    """
    draws = pgr_plr(weights @ counts)
    return _summarize_draws(pgr_plr(counts.sum(0)), draws), draws


def bootstrap_naive(counts: np.ndarray, n_boot: int,
                    rng: np.random.Generator) -> tuple[dict, dict]:
    """Transaction-level bootstrap that wrongly treats observations as i.i.d.

    Every (account, day, position) observation is resampled independently,
    which is a multinomial draw of the total number of observations over the
    four categories.

    Args:
        counts: (N, 4) per-account counts (integer conventions only).
        n_boot: Number of replications.
        rng: Generator.

    Returns:
        ``(summary, draws)`` as in :func:`bootstrap_accounts`.
    """
    total = counts.sum(0)
    m = int(round(total.sum()))
    draws = pgr_plr(rng.multinomial(m, total / total.sum(), size=n_boot))
    return _summarize_draws(pgr_plr(total), draws), draws


def account_panel(sim: SimulationResult, market: Market) -> pd.DataFrame:
    """One row per account with every variable of the turnover regressions.

    Returns are annualized arithmetic means of daily returns in percent, so
    ``net = gross - costs`` holds exactly account by account. ``gross`` is
    marked at mid-quotes (gross of the spread); ``gross_fill`` marks trades at
    their fill prices, i.e. the definitional error that leaves half the spread
    inside the "gross" return. Turnover is half of purchases plus sales over
    average wealth, annualized (the 1/2 sum|dw| of the mean-variance lecture);
    the endowment at day 0 is not trading.

    Args:
        sim: Simulation output.
        market: Its price universe (for the market-factor beta).

    Returns:
        DataFrame indexed by account.
    """
    g, v, pop = sim.gross, sim.value_prev, sim.population
    ann = TRADING_DAYS * 100.0
    mkt = market.market_returns - market.market_returns.mean()
    return pd.DataFrame({
        "gross": g.mean(0) * ann,
        "net": sim.net.mean(0) * ann,
        "costs": ((sim.commission + sim.spread) / v).mean(0) * ann,
        "commission": (sim.commission / v).mean(0) * ann,
        "spread": (sim.spread / v).mean(0) * ann,
        "gross_fill": (g - sim.spread / v).mean(0) * ann,
        "gross_log": np.log1p(g).mean(0) * ann,
        "net_log": np.log1p(sim.net).mean(0) * ann,
        "turnover": 0.5 * (sim.buy_value + sim.sell_value).sum(0) / v.mean(0)
        * TRADING_DAYS / g.shape[0],
        "beta": mkt @ (g - g.mean(0)) / (mkt @ mkt),
        "vol": g.std(0) * np.sqrt(TRADING_DAYS) * 100.0,
        "log_wealth": np.log(pop.wealth),
        "n_positions": pop.n_positions,
        "delta": pop.delta,
        "kappa": pop.kappa,
        "h0_annual": pop.h0 * TRADING_DAYS,
        "cash_share": sim.cash_share,
    })


CONTROLS = ("log_wealth", "n_positions", "beta")
OUTCOMES = ("gross", "net", "costs", "gross_fill", "gross_log", "net_log")


def _hc1(xhat: np.ndarray, resid: np.ndarray, bread: np.ndarray) -> np.ndarray:
    """HC1 sandwich ``n/(n-k) B^-1 (X' diag(e^2) X) B^-1`` for each outcome."""
    n, k = xhat.shape
    se = []
    for e in resid.T:
        meat = (xhat * e[:, None] ** 2).T @ xhat
        se.append(np.sqrt(np.diag(bread @ meat @ bread) * n / (n - k)))
    return np.array(se).T


def fit_linear(y: np.ndarray, x: np.ndarray, z: np.ndarray | None = None,
               weights: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """OLS (``z is None``) or 2SLS, for several outcomes and bootstrap weights.

    Args:
        y: (n, m) outcomes.
        x: (n, k) regressors including the constant; column 1 is turnover.
        z: (n, k) instruments (``x`` with turnover replaced), or None.
        weights: Optional (B, n) account-bootstrap multiplicities; when given
            the fit is repeated for every row (weighted least squares with
            integer weights is identical to resampling rows).

    Returns:
        Dict with ``coef`` (k, m) and ``se_hc1`` (k, m) for the unweighted fit,
        plus ``boot`` (B, k, m) replicate coefficients when weights are given.
    """
    z = x if z is None else z
    ztz_inv = np.linalg.inv(z.T @ z)
    xhat = z @ (ztz_inv @ (z.T @ x))
    bread = np.linalg.inv(xhat.T @ x)
    coef = bread @ (xhat.T @ y)
    out = {"coef": coef, "se_hc1": _hc1(xhat, y - x @ coef, np.linalg.inv(xhat.T @ xhat))}
    if weights is not None:
        w = weights.astype(float)
        pi = np.linalg.solve(np.einsum("bn,ni,nj->bij", w, z, z),
                             np.einsum("bn,ni,nj->bij", w, z, x))
        xh = np.einsum("ni,bij->bnj", z, pi)
        out["boot"] = np.linalg.solve(np.einsum("bn,bni,nj->bij", w, xh, x),
                                      np.einsum("bn,bni,nm->bim", w, xh, y))
    return out


def turnover_slopes(panel: pd.DataFrame, weights: np.ndarray,
                    instrument: str = "h0_annual") -> pd.DataFrame:
    """Barber-Odean regressions of every outcome on turnover, OLS and 2SLS.

    ``r_i = a + b * Turnover_i + g' X_i + e_i`` with X = log starting wealth,
    number of positions and the account's market beta. The IV column
    instruments realized turnover with the ex-ante baseline hazard implied by
    the agent's parameters (``h0 * 252``), which returns cannot cause.

    Args:
        panel: Output of :func:`account_panel`.
        weights: (B, N) account-bootstrap multiplicities.
        instrument: Panel column used as the excluded instrument.

    Returns:
        Long DataFrame: method, outcome, slope, HC1 SE, bootstrap SE and 95 %
        percentile CI (pp of annual return per unit of annual turnover), N,
        and the first-stage F for the IV rows.
    """
    ones = np.ones(len(panel))
    ctrl = panel[list(CONTROLS)].to_numpy()
    x = np.column_stack([ones, panel["turnover"], ctrl])
    z = np.column_stack([ones, panel[instrument], ctrl])
    y = panel[list(OUTCOMES)].to_numpy()
    first = fit_linear(panel[["turnover"]].to_numpy(), z)
    f_stat = float((first["coef"][1, 0] / first["se_hc1"][1, 0]) ** 2)
    rows = []
    for method, instr in (("OLS", None), ("IV", z)):
        fit = fit_linear(y, x, instr, weights)
        boot = fit["boot"][:, 1, :]
        for j, name in enumerate(OUTCOMES):
            lo, hi = np.quantile(boot[:, j], [0.025, 0.975])
            rows.append({"method": method, "outcome": name, "slope": fit["coef"][1, j],
                         "se_hc1": fit["se_hc1"][1, j], "se_boot": boot[:, j].std(ddof=1),
                         "ci_lo": lo, "ci_hi": hi, "n_accounts": len(panel),
                         "first_stage_F": f_stat if method == "IV" else np.nan})
    return pd.DataFrame(rows)


def _abnormal_forward(market: Market, horizon: int, adjust: bool) -> np.ndarray:
    """(T+1, S) forward returns net of the same-day cross-sectional mean.

    With ``adjust`` the market component ``beta_s * market forward return`` is
    removed first, using the true loadings (a ground-truth risk adjustment).
    """
    fwd = forward_returns(market.prices, horizon)
    if adjust:
        growth = np.concatenate([[1.0], np.cumprod(1.0 + market.market_returns)])
        cum = np.full(growth.shape, np.nan)
        cum[:-horizon] = growth[horizon:] / growth[:-horizon] - 1.0
        fwd = fwd - cum[:, None] * market.betas
    fwd[:-horizon] -= fwd[:-horizon].mean(axis=1, keepdims=True)
    return fwd


def _nw_se(score: np.ndarray, lags: int) -> float:
    """Newey-West (Bartlett) standard deviation of a sum of daily scores."""
    var = score @ score
    for lag in range(1, lags + 1):
        var += 2.0 * (1.0 - lag / (lags + 1.0)) * (score[lag:] @ score[:-lag])
    return float(np.sqrt(var))


def leakage_test(events: list[tuple[np.ndarray, np.ndarray]], market: Market,
                 horizon: int, adjust: bool) -> dict[str, float]:
    """Do bought securities have abnormal forward returns? (should be zero).

    Pooled over every purchase of every account and scenario given, this is
    the regression of the realized ``horizon``-day forward return on a buy
    indicator with day fixed effects (the control group being every security
    on the same day). Overlapping windows make the daily series dependent,
    so the SE is Newey-West with ``horizon`` lags on daily score sums.

    Args:
        events: List of ``(day, security)`` arrays, one pair per run pooled.
        market: The common price universe of those runs.
        horizon: Forward window in days.
        adjust: Remove the beta-scaled market forward return first.

    Returns:
        Dict with the coefficient and SE in percentage points, t, p-value,
        number of purchases and of distinct days.
    """
    ab = _abnormal_forward(market, horizon, adjust)
    day = np.concatenate([e[0] for e in events]).astype(np.int64)
    sec = np.concatenate([e[1] for e in events]).astype(np.int64)
    y = ab[day, sec]
    ok = np.isfinite(y)
    day, y = day[ok], y[ok]
    theta = y.mean()
    score = np.bincount(day, y - theta, minlength=ab.shape[0])
    se = _nw_se(score, horizon) / y.size
    t = theta / se
    return {"horizon": horizon, "adjusted": adjust, "coef_pp": 100 * theta,
            "se_pp": 100 * se, "t": t, "p": 2 * stats.norm.sf(abs(t)),
            "n_buys": int(y.size), "n_days": int(np.unique(day).size)}


def bought_minus_sold(buys: tuple[np.ndarray, np.ndarray],
                      sales: tuple[np.ndarray, np.ndarray], market: Market,
                      horizon: int = 252, adjust: bool = False) -> dict[str, float]:
    """Forward return of securities bought minus securities sold (Odean 1999).

    Args:
        buys, sales: ``(day, security)`` arrays of purchases and sales.
        market: Price universe.
        horizon: Forward window (252 = one year, Odean's horizon).
        adjust: Remove the beta-scaled market forward return first.

    Returns:
        Dict with the difference and its Newey-West SE in pp, t and p-value.
    """
    ab = _abnormal_forward(market, horizon, adjust)
    parts = []
    for day, sec in (buys, sales):
        y = ab[day.astype(np.int64), sec.astype(np.int64)]
        ok = np.isfinite(y)
        parts.append((day[ok].astype(np.int64), y[ok]))
    (db, yb), (ds, ys) = parts
    diff = yb.mean() - ys.mean()
    n_days = ab.shape[0]
    score = (np.bincount(db, yb - yb.mean(), n_days) / yb.size
             - np.bincount(ds, ys - ys.mean(), n_days) / ys.size)
    se = _nw_se(score, horizon)
    return {"horizon": horizon, "adjusted": adjust, "diff_pp": 100 * diff,
            "se_pp": 100 * se, "t": diff / se, "p": 2 * stats.norm.sf(abs(diff / se)),
            "n_buys": int(yb.size), "n_sales": int(ys.size)}


def holding_periods(sim: SimulationResult, n_bins: int = 10) -> pd.DataFrame:
    """Completed holding periods of winners vs losers by decile of delta.

    Winners/losers are classified against the average cost at the sale; the
    share of positions still open at the end (right-censored) is reported per
    domain because strong disposition keeps losers open past the sample.

    Args:
        sim: Simulation output (``sales`` and ``open_end`` tables).
        n_bins: Number of quantile bins of delta.

    Returns:
        DataFrame by delta bin with mean holding days of realized winners and
        losers, their ratio, event counts and censored shares.
    """
    delta = sim.population.delta
    edges = np.quantile(delta, np.linspace(0, 1, n_bins + 1))
    rows = []
    sales, open_end = sim.tables["sales"], sim.tables["open_end"]
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        in_bin = (delta >= lo) & ((delta <= hi) if b == n_bins - 1 else (delta < hi))
        s_mask, o_mask = in_bin[sales["account"]], in_bin[open_end["account"]]
        win, lose = sales["ret_vs_cost"] > 0, sales["ret_vs_cost"] < 0
        o_win, o_lose = open_end["ret_vs_cost"] > 0, open_end["ret_vs_cost"] < 0
        hw = sales["hold_days"][s_mask & win].mean()
        hl = sales["hold_days"][s_mask & lose].mean()
        rows.append({"delta_lo": lo, "delta_hi": hi, "hold_winners": hw, "hold_losers": hl,
                     "ratio_losers_winners": hl / hw, "n_sold_winners": int((s_mask & win).sum()),
                     "n_sold_losers": int((s_mask & lose).sum()),
                     "open_winners": int((o_mask & o_win).sum()),
                     "open_losers": int((o_mask & o_lose).sum())})
    return pd.DataFrame(rows)


def corr_ci(x: np.ndarray, y: np.ndarray, alpha: float = 0.05) -> dict[str, float]:
    """Pearson correlation with a Fisher-z confidence interval.

    Args:
        x, y: Samples of equal length.
        alpha: One minus the confidence level.

    Returns:
        Dict with r, CI bounds, p-value and n.
    """
    r, p = stats.pearsonr(x, y)
    z, half = np.arctanh(r), stats.norm.ppf(1 - alpha / 2) / np.sqrt(len(x) - 3)
    return {"r": float(r), "lo": float(np.tanh(z - half)), "hi": float(np.tanh(z + half)),
            "p": float(p), "n": len(x)}


def sale_day_profile(sim: SimulationResult) -> dict[str, float]:
    """Composition of the account-days that enter the Odean counts.

    Args:
        sim: Simulation output (``snapshots`` table).

    Returns:
        Dict with the number of sale days, positions observed and sold per
        sale day, the share of sale days with two or more sales, the shares
        of paper gains and losses among the observations (average-cost
        classification) and the share of positions holding several lots.
    """
    snap = sim.tables["snapshots"]
    key = snap["account"].astype(np.int64) * (int(snap["day"].max()) + 1) + snap["day"]
    uk, inv = np.unique(key, return_inverse=True)
    sold = np.bincount(inv, snap["shares_sold"] > 0)
    sign = classify(snap["mid"], snap["avg_cost"])
    return {"sale_days": int(uk.size), "positions_per_sale_day": snap["account"].size / uk.size,
            "sold_per_sale_day": float(sold.mean()),
            "share_multi_sale_days": float((sold >= 2).mean()),
            "share_paper_gains": float((sign == 1).mean()),
            "share_paper_losses": float((sign == -1).mean()),
            "share_multi_lot": float((snap["n_lots"] > 1).mean())}


def tiebreak_profile(sim: SimulationResult, market: Market, window: int = 40,
                     threshold: float = 0.12, band: float = 0.25) -> dict[str, float]:
    """Realization rates on sale days split by data PGR/PLR throws away.

    Three splits that separate mechanisms producing the same four counts:
    sign vs cost crossed with the trailing return (a reversal believer reacts
    to the trailing move, a disposition investor to the purchase price);
    sign crossed with the position's weight relative to equal weight (a
    rebalancer reacts to weight); and the share of realized gains that are
    trims rather than full exits (order-level data).

    Args:
        sim: Simulation output (``snapshots`` table).
        market: Price universe (for trailing returns).
        window, threshold: Trailing-return split (same values as the belief).
        band: Overweight split, weight above ``(1 + band)/n``.

    Returns:
        Dict of realization rates (count convention) and the trim share.
    """
    snap = sim.tables["snapshots"]
    day, sec = snap["day"].astype(np.int64), snap["security"].astype(np.int64)
    trail = market.prices[day, sec] / market.prices[np.maximum(day - window, 0), sec] - 1.0
    hot = trail > threshold
    value = snap["shares"] * snap["mid"]
    key = snap["account"].astype(np.int64) * (int(day.max()) + 1) + day
    _, inv = np.unique(key, return_inverse=True)
    weight_x_n = value / np.bincount(inv, value)[inv] * sim.population.n_positions[snap["account"]]
    over = weight_x_n > 1.0 + band
    sign = classify(snap["mid"], snap["avg_cost"])
    realized = snap["shares_sold"] > 0
    out = {}
    for tag, split in (("hot", hot), ("over", over)):
        for s_name, s_val in (("gain", 1), ("loss", -1)):
            for flag, f_name in ((True, tag), (False, f"not_{tag}")):
                m = (sign == s_val) & (split == flag)
                out[f"rate_{s_name}_{f_name}"] = float(realized[m].mean()) if m.any() else np.nan
    gains_realized = realized & (sign == 1)
    out["trim_share_of_gain_realizations"] = float((snap["kind"][gains_realized] >= 2).mean())
    return out


def summarize_run(sim: SimulationResult, market: Market, n_boot: int,
                  rng: np.random.Generator) -> dict:
    """Apply every estimator to one run and drop the bulky raw tables.

    Args:
        sim: Simulation output.
        market: Its price universe.
        n_boot: Account-bootstrap replications (shared by both estimators).
        rng: Generator for the bootstrap.

    Returns:
        Dict with per-convention counts and bootstraps, the naive bootstrap,
        the account panel, the slope table, the purchase/sale event arrays
        needed by the leakage test and the runtime.
    """
    weights = account_weights(sim.population.n_agents, n_boot, rng)
    conv, draws = {}, {}
    for name in CONVENTIONS:
        dc = disposition_counts(sim, name)
        summary, draws[name] = bootstrap_accounts(dc["counts"], weights)
        conv[name] = {**summary, "counts": dc["counts"], "ties": dc["ties"],
                      "observations": dc["observations"]}
    naive, naive_draws = bootstrap_naive(conv["avg_count"]["counts"], n_boot, rng)
    panel = account_panel(sim, market)
    buys, sales = sim.tables["buys"], sim.tables["sales"]
    return {"scenario": sim.scenario, "conventions": conv, "draws": draws,
            "naive": naive, "naive_draws": naive_draws, "panel": panel,
            "profile": sale_day_profile(sim), "tiebreak": tiebreak_profile(sim, market),
            "slopes": turnover_slopes(panel, weights),
            "buys": (buys["day"], buys["security"]), "sales": (sales["day"], sales["security"]),
            "trade_values": np.concatenate([buys["value"], sales["value"]]),
            "sale_hold": (sales["hold_days"], sales["ret_vs_cost"] > 0),
            "runtime_s": sim.runtime_s}
