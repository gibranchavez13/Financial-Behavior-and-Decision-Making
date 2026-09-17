"""Price universe for the behavioral-bias simulator.

The whole price path is generated once, before any agent exists, from its own
random stream. Nothing an agent does can feed back into it, which is the
structural half of the "no informed trading" constraint (the statistical half
is the leakage test in ``simlib_estimadores``).

Model: one priced market factor plus ``k`` orthogonal, zero-premium style
factors, heterogeneous loadings and idiosyncratic noise, with CAPM-consistent
drifts (zero alpha for every security).
"""

# ======================================================================================
# PRE-ANALYSIS (registered 2026-09-10T23:17:59Z, before any estimator existed; never
# edited). Official text: resultados/pre_analisis.md (SHA-256 df22412ecfa7b292...).
# This copy renders its LaTeX as plain characters; the wording is unchanged.
# Role of this file: the price universe these expectations are conditioned on (100
#   securities, 756 days, zero alpha; the realized market returned 2.1 %/yr).
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

from dataclasses import dataclass

import numpy as np

TRADING_DAYS = 252


@dataclass(frozen=True)
class MarketConfig:
    """Calibration of the price universe (annualized where applicable).

    Attributes:
        n_securities: Number of securities S.
        n_days: Number of trading days T (prices exist for days 0..T).
        mu_market: Expected annual (arithmetic) return of the market factor.
        sigma_market: Annual volatility of the market factor.
        n_style: Number of orthogonal style factors k (zero premium).
        sigma_style: Annual volatility of each style factor.
        beta_mean, beta_sd, beta_min, beta_max: Distribution of market betas.
        idio_vol_low, idio_vol_high: Uniform range of idiosyncratic volatility.
        price0_low, price0_high: Log-uniform range of initial prices (USD).
        tick: Quote grid in USD; mid-quotes are rounded to it.
        gbm: If True, switch off all factors (i.i.d. GBM-like baseline with a
            common drift), kept as the documented fallback model.
    """

    n_securities: int = 100
    n_days: int = 756
    mu_market: float = 0.07
    sigma_market: float = 0.16
    n_style: int = 3
    sigma_style: float = 0.05
    beta_mean: float = 1.0
    beta_sd: float = 0.30
    beta_min: float = 0.30
    beta_max: float = 1.80
    idio_vol_low: float = 0.20
    idio_vol_high: float = 0.40
    price0_low: float = 10.0
    price0_high: float = 150.0
    tick: float = 0.01
    gbm: bool = False


@dataclass(frozen=True)
class Market:
    """A realized price universe.

    Attributes:
        prices: (T+1, S) mid-quotes on the tick grid, day 0 included.
        returns: (T, S) simple returns of the quotes, row t is day t-1 -> t.
        factor_returns: (T, 1+k) daily factor returns; column 0 is the market
            factor *including* its drift, the rest are style factors.
        betas: (S,) market loadings.
        loadings: (S, k) style loadings.
        idio_vol: (S,) annual idiosyncratic volatility.
        mu: (S,) expected annual return, equal to ``betas * mu_market``.
        config: The calibration that produced it.
    """

    prices: np.ndarray
    returns: np.ndarray
    factor_returns: np.ndarray
    betas: np.ndarray
    loadings: np.ndarray
    idio_vol: np.ndarray
    mu: np.ndarray
    config: MarketConfig

    @property
    def market_returns(self) -> np.ndarray:
        """(T,) daily return of the market factor (drift included)."""
        return self.factor_returns[:, 0]


def simulate_market(cfg: MarketConfig, rng: np.random.Generator) -> Market:
    """Draw loadings, factor paths and idiosyncratic shocks; build quotes.

    Simple daily returns are
    ``r_st = beta_s*(mu_M/252 + f_Mt) + b_s' f_t + e_st`` with Gaussian shocks,
    so the expected annual return of every security is exactly
    ``beta_s * mu_M``: expected returns differ only through priced risk and
    every alpha is zero. A latent price compounds these returns and the
    tradable mid-quote is the latent price rounded to the tick.

    Args:
        cfg: Market calibration.
        rng: Generator reserved for the market; agents never touch it.

    Returns:
        The realized :class:`Market`.
    """
    S, T, k = cfg.n_securities, cfg.n_days, cfg.n_style
    dt = 1.0 / TRADING_DAYS
    if cfg.gbm:
        betas = np.ones(S)
        loadings = np.zeros((S, k))
    else:
        betas = np.clip(rng.normal(cfg.beta_mean, cfg.beta_sd, S),
                        cfg.beta_min, cfg.beta_max)
        loadings = rng.normal(0.0, 1.0, (S, k))
    idio_vol = rng.uniform(cfg.idio_vol_low, cfg.idio_vol_high, S)

    mkt_sigma = 0.0 if cfg.gbm else cfg.sigma_market
    mkt = cfg.mu_market * dt + rng.normal(0.0, mkt_sigma * np.sqrt(dt), T)
    style = rng.normal(0.0, cfg.sigma_style * np.sqrt(dt), (T, k))
    eps = rng.normal(0.0, 1.0, (T, S)) * (idio_vol * np.sqrt(dt))
    if cfg.gbm:
        # Common drift, no common shocks: total volatility = idiosyncratic.
        latent_ret = cfg.mu_market * dt + eps
    else:
        latent_ret = mkt[:, None] * betas + style @ loadings.T + eps

    p0 = np.exp(rng.uniform(np.log(cfg.price0_low), np.log(cfg.price0_high), S))
    latent = p0 * np.vstack([np.ones(S), np.cumprod(1.0 + latent_ret, axis=0)])
    prices = np.maximum(np.round(latent / cfg.tick) * cfg.tick, cfg.tick)
    returns = prices[1:] / prices[:-1] - 1.0
    if cfg.gbm:
        # No common factor exists; the equal-weighted index stands in for the
        # market so that portfolio betas remain defined.
        mkt = latent_ret.mean(axis=1)
    factor_returns = np.column_stack([mkt, style])
    return Market(prices=prices, returns=returns, factor_returns=factor_returns,
                  betas=betas, loadings=loadings, idio_vol=idio_vol,
                  mu=betas * cfg.mu_market, config=cfg)


def implied_covariance(market: Market) -> np.ndarray:
    """Annual covariance matrix implied by the factor model.

    ``Sigma = s_M^2 beta beta' + sigma_style^2 B B' + diag(idio_vol^2)``.

    Args:
        market: A realized market (only its loadings and calibration are used).

    Returns:
        (S, S) covariance matrix of annual returns.
    """
    cfg = market.config
    s_m = 0.0 if cfg.gbm else cfg.sigma_market
    cov = s_m ** 2 * np.outer(market.betas, market.betas)
    cov += cfg.sigma_style ** 2 * market.loadings @ market.loadings.T
    cov += np.diag(market.idio_vol ** 2)
    return cov


def correlation_summary(market: Market) -> dict[str, float]:
    """Implied vs realized average pairwise correlation and volatility.

    Args:
        market: A realized market.

    Returns:
        Dict with the mean off-diagonal implied and realized correlations, the
        mean implied and realized annual volatility, and the realized annual
        mean and volatility of the market factor.
    """
    cov = implied_covariance(market)
    vol = np.sqrt(np.diag(cov))
    corr = cov / np.outer(vol, vol)
    off = ~np.eye(len(vol), dtype=bool)
    realized = np.corrcoef(market.returns.T)
    mkt = market.market_returns
    return {
        "corr_implied_mean": float(corr[off].mean()),
        "corr_realized_mean": float(realized[off].mean()),
        "vol_implied_mean": float(vol.mean()),
        "vol_realized_mean": float(market.returns.std(axis=0).mean()
                                   * np.sqrt(TRADING_DAYS)),
        "market_mean_realized": float(mkt.mean() * TRADING_DAYS),
        "market_vol_realized": float(mkt.std() * np.sqrt(TRADING_DAYS)),
    }


def realized_reversal(market: Market, window: int = 126, step: int = 21) -> float:
    """Realized cross-sectional reversal of one path (a property of luck, not of the DGP).

    Average cross-sectional correlation between each security's past and next
    ``window``-day residual log return (market component removed with the true
    betas), sampled every ``step`` days. The DGP has no reversal, so across
    paths this averages zero; a negative value on a given path means past
    relative winners happened to lag afterwards, which pays churning and
    rebalancing *ex post* without anyone having information.

    Args:
        market: A realized market.
        window: Look-back and look-ahead length in days.
        step: Sampling interval in days.

    Returns:
        The average correlation.
    """
    lr = np.log1p(market.returns) - np.outer(np.log1p(market.market_returns), market.betas)
    cum = np.vstack([np.zeros(lr.shape[1]), np.cumsum(lr, axis=0)])
    days = range(window, lr.shape[0] - window + 1, step)
    return float(np.mean([np.corrcoef(cum[t] - cum[t - window], cum[t + window] - cum[t])[0, 1]
                          for t in days]))


def forward_returns(prices: np.ndarray, horizon: int) -> np.ndarray:
    """Realized simple return from day t to day t+horizon for every security.

    Args:
        prices: (T+1, S) quotes.
        horizon: Forward window in trading days.

    Returns:
        (T+1, S) array; rows without a full forward window are NaN.
    """
    out = np.full(prices.shape, np.nan)
    out[:-horizon] = prices[horizon:] / prices[:-horizon] - 1.0
    return out
