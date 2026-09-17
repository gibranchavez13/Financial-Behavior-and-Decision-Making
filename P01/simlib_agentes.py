"""Agent population and trading engine for the behavioral-bias simulator.

Injected mechanism (A, domain-dependent holding hazard): every day each held
position is sold with probability ``h0_i * (1 + delta_i)`` if it trades above
the agent's reference point (average cost of the open lots) and
``h0_i * (1 - delta_i)`` if it trades below. ``h0_i`` grows with ``kappa_i``.
PGR, PLR and turnover are never set: they emerge from these hazards acting on
a price path the agents cannot see ahead of time.
"""

# ======================================================================================
# PRE-ANALYSIS (registered 2026-09-10T23:17:59Z, before any estimator existed; never
# edited). Official text: resultados/pre_analisis.md (SHA-256 df22412ecfa7b292...).
# This copy renders its LaTeX as plain characters; the wording is unchanged.
# Role of this file: implements the injected parameters (delta, kappa, h0 = l(1 + 3 kappa)/252,
#   the two confounds) and the trading they produce.
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

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

from simlib_precios import TRADING_DAYS, Market

TIE_USD = 0.005  # half a cent: |mid - reference| below this is a tie
_EPS_SH = 1e-9   # shares below this are treated as zero


@dataclass(frozen=True)
class BehaviorConfig:
    """Decision-rule parameters shared by every scenario.

    Attributes:
        wealth_low, wealth_high: Log-uniform range of starting capital (USD).
        n_pos_low, n_pos_high: Integer-uniform range of positions held.
        base_rate_low, base_rate_high: Uniform range of the annual baseline
            sale intensity per position (liquidity needs, independent of
            delta, kappa and prices).
        kappa_scale: ``h0 = base_rate * (1 + kappa_scale * kappa) / 252``.
        partial_prob: Probability that a triggered sale is a trim, not an exit.
        partial_fraction: Share of the position sold in a trim.
        rebalance_band: Rebalancers trim when weight > (1 + band) / n.
        mr_window: Trailing window (days) of the mean-reversion belief.
        mr_threshold: Trailing return above which a security is "too high".
        mr_multiplier: Hazard on such securities is ``h0 * (1 + multiplier)``.
        max_lots: Lots kept per position; extra purchases merge into the last.
        min_trade_usd: Trims smaller than this are not worth a ticket: a
            rebalancing trim below it is skipped and a random partial sale
            below it becomes a full exit.
    """

    wealth_low: float = 10_000.0
    wealth_high: float = 500_000.0
    n_pos_low: int = 5
    n_pos_high: int = 30
    base_rate_low: float = 0.25
    base_rate_high: float = 1.25
    kappa_scale: float = 3.0
    partial_prob: float = 0.25
    partial_fraction: float = 0.5
    rebalance_band: float = 0.25
    mr_window: int = 40
    mr_threshold: float = 0.12
    mr_multiplier: float = 4.0
    max_lots: int = 8
    min_trade_usd: float = 100.0


@dataclass(frozen=True)
class CostConfig:
    """Explicit trading costs.

    Attributes:
        commission_usd: Flat commission per ticket (every buy and every sell).
        spread_bps: Full quoted bid-ask spread in basis points of the mid.
            Buys fill at ``mid*(1+s/2)``, sells at ``mid*(1-s/2)``.
    """

    commission_usd: float = 10.0
    spread_bps: float = 20.0

    @property
    def half_spread(self) -> float:
        """Half spread as a fraction of the mid-quote."""
        return self.spread_bps / 2.0 / 1e4


@dataclass(frozen=True)
class ScenarioConfig:
    """One cell of the scenario grid.

    Attributes:
        key: Short identifier (e.g. ``"S3"``).
        label: Human-readable name.
        delta: Population constant for delta, or None for U(0, 1) draws.
        kappa: Population constant for kappa, or None for U(0, 1) draws.
        confound: ``"none"``, ``"rebalancing"`` or ``"mean_reversion"``.
        n_agents: Number of accounts N.
        cash_lag_days: Days sale proceeds sit in cash before redeployment.
        corr_delta_kappa: Target Pearson correlation of the U(0, 1) marginals
            (Gaussian copula); 0 means independent streams. Heterogeneous only.
    """

    key: str
    label: str
    delta: float | None
    kappa: float | None
    confound: str = "none"
    n_agents: int = 1000
    cash_lag_days: int = 0
    corr_delta_kappa: float = 0.0


@dataclass(frozen=True)
class Population:
    """Account-level primitives (all arrays have shape (N,)).

    Attributes:
        wealth: Starting capital W_i (USD).
        n_positions: Positions held n_i.
        delta: Disposition strength delta_i.
        kappa: Overprecision / churn intensity kappa_i.
        base_rate: Annual baseline sale intensity per position.
        h0: Daily baseline hazard ``base_rate*(1+kappa_scale*kappa)/252``.
    """

    wealth: np.ndarray
    n_positions: np.ndarray
    delta: np.ndarray
    kappa: np.ndarray
    base_rate: np.ndarray
    h0: np.ndarray

    @property
    def n_agents(self) -> int:
        """Number of accounts."""
        return len(self.wealth)


def child_seed(seed: np.random.SeedSequence, *key: int) -> np.random.SeedSequence:
    """Deterministic sub-stream of ``seed`` addressed by an integer key.

    Unlike ``SeedSequence.spawn`` this does not depend on how many children
    were spawned before, so every stream is individually reproducible.

    Args:
        seed: Parent seed sequence.
        *key: Integers appended to the parent's spawn key.

    Returns:
        The child seed sequence.
    """
    return np.random.SeedSequence(seed.entropy, spawn_key=seed.spawn_key + key)


def draw_population(scn: ScenarioConfig, beh: BehaviorConfig,
                    seed: np.random.SeedSequence) -> Population:
    """Draw the account population of one scenario.

    Each primitive comes from its own sub-stream, so delta and kappa are
    independent *by construction* whenever both vary: they never share a
    generator. The Gaussian copula is used only when a correlation is asked
    for on purpose (the deliberately mis-specified run of Section 10).

    Args:
        scn: Scenario (constants or None for heterogeneous draws).
        beh: Behavioral calibration.
        seed: Scenario seed; sub-streams 0-4 are used here.

    Returns:
        The :class:`Population`.
    """
    n = scn.n_agents
    g = [np.random.default_rng(child_seed(seed, 0, j)) for j in range(5)]
    wealth = np.exp(g[0].uniform(np.log(beh.wealth_low), np.log(beh.wealth_high), n))
    n_pos = g[1].integers(beh.n_pos_low, beh.n_pos_high + 1, n)
    base = g[2].uniform(beh.base_rate_low, beh.base_rate_high, n)
    if scn.corr_delta_kappa:
        # Pearson corr of Phi(z1), Phi(z2) is (6/pi) asin(rho/2).
        rho = 2.0 * np.sin(np.pi * scn.corr_delta_kappa / 6.0)
        z = g[3].multivariate_normal([0.0, 0.0], [[1.0, rho], [rho, 1.0]], n)
        u_delta, u_kappa = norm.cdf(z[:, 0]), norm.cdf(z[:, 1])
    else:
        u_delta, u_kappa = g[3].uniform(0.0, 1.0, n), g[4].uniform(0.0, 1.0, n)
    delta = u_delta if scn.delta is None else np.full(n, float(scn.delta))
    kappa = u_kappa if scn.kappa is None else np.full(n, float(scn.kappa))
    h0 = base * (1.0 + beh.kappa_scale * kappa) / TRADING_DAYS
    return Population(wealth=wealth, n_positions=n_pos, delta=delta,
                      kappa=kappa, base_rate=base, h0=h0)


class _Book:
    """Mutable portfolio state of every account.

    Positions are (account, security) cells; each keeps up to ``max_lots``
    FIFO-ordered lots (lot 0 is the oldest). Sale proceeds wait in four
    ring-buffered pools until redeployed: cash for new positions, the number
    of positions to open, cash for a random top-up and cash for the most
    underweight position. With ``lag = 0`` the buffer has a single slot that
    is filled and emptied on the same day.
    """

    def __init__(self, n_agents: int, n_sec: int, max_lots: int, lag: int) -> None:
        shape = (n_agents, n_sec, max_lots)
        self.lot_sh = np.zeros(shape)
        self.lot_px = np.zeros(shape)
        self.lot_day = np.zeros(shape, dtype=np.int32)
        self.n_lots = np.zeros((n_agents, n_sec), dtype=np.int16)
        self.pos_sh = np.zeros((n_agents, n_sec))
        self.pos_cost = np.zeros((n_agents, n_sec))
        self.cash = np.zeros(n_agents)
        self.pools = np.zeros((4, n_agents, lag + 1))
        self.lag = lag

    def held(self) -> np.ndarray:
        """(N, S) mask of open positions."""
        return self.pos_sh > _EPS_SH

    def reference(self) -> np.ndarray:
        """(N, S) average cost of the open lots (0 where nothing is held)."""
        return np.divide(self.pos_cost, self.pos_sh, out=np.zeros_like(self.pos_cost),
                         where=self.held())

    def wealth(self, prices: np.ndarray) -> np.ndarray:
        """(N,) holdings at mid plus idle cash plus undeployed proceeds."""
        return (self.pos_sh * prices).sum(1) + self.cash + self.pools[[0, 2, 3]].sum((0, 2))


class _Recorder:
    """Collects per-day record arrays and concatenates them at the end."""

    def __init__(self) -> None:
        self.tables: dict[str, list[dict[str, np.ndarray]]] = {}

    def add(self, table: str, **cols: np.ndarray) -> None:
        """Append one day's batch of rows to ``table``."""
        self.tables.setdefault(table, []).append(cols)

    def finish(self, table: str) -> dict[str, np.ndarray]:
        """Concatenate a table's batches column by column (empty if unused)."""
        batches = self.tables.get(table, [])
        if not batches:
            return {}
        return {c: np.concatenate([b[c] for b in batches]) for c in batches[0]}


def _initial_book(pop: Population, prices0: np.ndarray, beh: BehaviorConfig,
                  lag: int, rng: np.random.Generator) -> _Book:
    """Endow each account with n_i distinct random securities, equal dollars.

    The endowment is bought at mid without costs and does not count as
    trading: turnover and costs measure only decisions taken during the sample.

    Args:
        pop: Account population.
        prices0: (S,) day-0 quotes.
        beh: Behavioral calibration (for ``max_lots``).
        lag: Cash-window length in days.
        rng: Generator of day 0.

    Returns:
        The initial :class:`_Book`.
    """
    n, s = pop.n_agents, len(prices0)
    book = _Book(n, s, beh.max_lots, lag)
    order = np.argsort(rng.random((n, s)), axis=1)
    chosen = np.arange(s)[None, :] < pop.n_positions[:, None]
    acc, rank = np.nonzero(chosen)
    sec = order[acc, rank]
    sh = pop.wealth[acc] / pop.n_positions[acc] / prices0[sec]
    book.lot_sh[acc, sec, 0] = sh
    book.lot_px[acc, sec, 0] = prices0[sec]
    book.n_lots[acc, sec] = 1
    book.pos_sh[acc, sec] = sh
    book.pos_cost[acc, sec] = sh * prices0[sec]
    return book


def _decide_sales(book: _Book, price: np.ndarray, trailing: np.ndarray,
                  pop: Population, scn: ScenarioConfig, beh: BehaviorConfig,
                  rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Apply the day's selling rules using only today's and past quotes.

    Kinds: 1 = full exit, 2 = random partial sale, 3 = rebalancing trim.
    The hazard multiplier is ``1 + delta`` above the reference point,
    ``1 - delta`` below it and 1 at a tie; the mean-reversion believer
    multiplies it by ``1 + mr_multiplier`` on securities whose trailing return
    exceeds the threshold (a belief about past moves, wrong by construction).

    Args:
        book: Current state (before today's trades).
        price: (S,) today's quotes.
        trailing: (S,) quotes ``mr_window`` days ago.
        pop, scn, beh: Population, scenario and calibration.
        rng: Generator of the day.

    Returns:
        ``(shares_sold, kind)``, both (N, S).
    """
    held = book.held()
    gap = price - book.reference()
    sign = np.where(gap > TIE_USD, 1.0, np.where(gap < -TIE_USD, -1.0, 0.0)) * held
    mult = 1.0 + pop.delta[:, None] * sign
    if scn.confound == "mean_reversion":
        hot = price / trailing - 1.0 > beh.mr_threshold
        mult = mult * (1.0 + beh.mr_multiplier * hot[None, :])
    sell = held & (rng.random(held.shape) < pop.h0[:, None] * mult)
    part = sell & (rng.random(held.shape) < beh.partial_prob)
    value = book.pos_sh * price
    part &= beh.partial_fraction * value >= beh.min_trade_usd
    kind = np.where(sell, np.where(part, 2, 1), 0).astype(np.int8)
    sold = np.where(kind == 1, book.pos_sh, 0.0)
    sold += np.where(kind == 2, beh.partial_fraction * book.pos_sh, 0.0)
    if scn.confound == "rebalancing":
        target = book.wealth(price) / pop.n_positions
        excess = value - target[:, None]
        over = held & (kind == 0) & (value > (1.0 + beh.rebalance_band) * target[:, None])
        over &= excess >= beh.min_trade_usd
        kind[over] = 3
        sold += np.where(over, excess / price, 0.0)
    return sold, kind


def _fifo_relief(lot_sh: np.ndarray, sold: np.ndarray, full: np.ndarray) -> np.ndarray:
    """Shares taken from each lot when ``sold`` shares leave, oldest first.

    Args:
        lot_sh: (M, L) open lot sizes, lot 0 oldest.
        sold: (M,) shares sold.
        full: (M,) True for full exits (every lot is relieved exactly).

    Returns:
        (M, L) relieved shares per lot.
    """
    cum_before = np.cumsum(lot_sh, axis=1) - lot_sh
    relief = np.clip(sold[:, None] - cum_before, 0.0, lot_sh)
    relief[full] = lot_sh[full]
    return relief


def _record_sale_day(book: _Book, rows: np.ndarray, price: np.ndarray, t: int,
                     sold: np.ndarray, kind: np.ndarray,
                     rec: _Recorder) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Write the brokerage snapshot of every account that sells today.

    Only the *state before today's trades* is recorded (what an Odean-style
    econometrician reconstructs from statements): each open position with its
    mid-quote, average, oldest and newest lot costs, shares held and sold,
    plus a lot-level table. Classification is left to the estimators.

    Args:
        book: State before trading.
        rows: Accounts with at least one sale today.
        price: (S,) today's quotes.
        t: Day index.
        sold, kind: Output of :func:`_decide_sales`.
        rec: Recorder.

    Returns:
        ``(account, security, relief)`` for every recorded position.
    """
    r, sec = np.nonzero(book.held()[rows])
    acc = rows[r]
    ls, px = book.lot_sh[acc, sec], book.lot_px[acc, sec]
    nl = book.n_lots[acc, sec].astype(np.int64)
    s_sold, k = sold[acc, sec], kind[acc, sec]
    relief = _fifo_relief(ls, s_sold, k == 1)
    day = np.full(acc.size, t, np.int16)
    rec.add("snapshots", account=acc.astype(np.int32), day=day,
            security=sec.astype(np.int16), mid=price[sec],
            avg_cost=book.pos_cost[acc, sec] / book.pos_sh[acc, sec],
            fifo_cost=px[:, 0], last_cost=px[np.arange(acc.size), nl - 1],
            shares=book.pos_sh[acc, sec].astype(np.float32),
            shares_sold=s_sold.astype(np.float32), n_lots=nl.astype(np.int8), kind=k)
    lr, ll = np.nonzero(ls > _EPS_SH)
    rec.add("lots", account=acc[lr].astype(np.int32), day=day[lr],
            mid=price[sec[lr]], cost=px[lr, ll],
            shares=ls[lr, ll].astype(np.float32),
            shares_sold=relief[lr, ll].astype(np.float32))
    return acc, sec, relief


def _execute_sales(book: _Book, acc: np.ndarray, sec: np.ndarray, relief: np.ndarray,
                   sold: np.ndarray, kind: np.ndarray, price: np.ndarray, t: int,
                   costs: CostConfig, rec: _Recorder) -> tuple[np.ndarray, np.ndarray]:
    """Relieve lots FIFO, book proceeds into the redeployment pools.

    Sells fill at ``mid*(1-s/2)`` and pay one commission per ticket. Proceeds
    of full exits fund new positions; those of random partial sales fund a
    random top-up; those of rebalancing trims fund the most underweight
    position. Each pool is released ``lag`` days later.

    Args:
        book: State (modified in place).
        acc, sec, relief: Snapshot pairs and their FIFO relief.
        sold, kind: (N, S) decisions of the day.
        price: (S,) quotes.
        t: Day index.
        costs: Cost calibration.
        rec: Recorder (``sales`` table).

    Returns:
        ``(sell_value_mid, n_sell_tickets)`` per account, both (N,).
    """
    s_sold, k = sold[acc, sec], kind[acc, sec]
    m = s_sold > 0
    acc, sec, relief, s_sold, k = acc[m], sec[m], relief[m], s_sold[m], k[m]
    dy = book.lot_day[acc, sec]
    avg = book.pos_cost[acc, sec] / book.pos_sh[acc, sec]
    left = book.lot_sh[acc, sec] - relief
    left[(left < _EPS_SH) | (k == 1)[:, None]] = 0.0
    order = np.argsort(left <= 0.0, axis=1, kind="stable")
    keep = np.take_along_axis(left, order, 1)
    alive = keep > 0.0
    book.lot_px[acc, sec] = np.take_along_axis(book.lot_px[acc, sec], order, 1) * alive
    book.lot_day[acc, sec] = np.take_along_axis(dy, order, 1) * alive
    book.lot_sh[acc, sec] = keep
    book.n_lots[acc, sec] = alive.sum(1)
    book.pos_sh[acc, sec] = keep.sum(1)
    book.pos_cost[acc, sec] = (keep * book.lot_px[acc, sec]).sum(1)

    value = s_sold * price[sec]
    proceeds = value * (1.0 - costs.half_spread) - costs.commission_usd
    hold = (relief * (t - dy)).sum(1) / np.maximum(relief.sum(1), _EPS_SH)
    rec.add("sales", account=acc.astype(np.int32), day=np.full(acc.size, t, np.int16),
            security=sec.astype(np.int16), kind=k, value=value,
            ret_vs_cost=price[sec] / avg - 1.0, hold_days=hold)
    n, slot = book.cash.size, (t + book.lag) % (book.lag + 1)
    for pool, code in ((0, 1), (2, 2), (3, 3)):
        book.pools[pool, :, slot] += np.bincount(acc[k == code], proceeds[k == code], n)
    book.pools[1, :, slot] += np.bincount(acc[k == 1], minlength=n)
    return np.bincount(acc, value, n), np.bincount(acc, minlength=n)


def _release_pools(book: _Book, price: np.ndarray, t: int, traded: np.ndarray,
                   rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    """Turn the pools due today into buy orders.

    New positions go to securities drawn uniformly among those the account
    neither holds nor traded today: the draw uses no price information at
    all, so it is orthogonal to future returns by construction. Top-ups go
    to a uniformly random held position, or (rebalancers) to the most
    underweight one; neither can be a position traded today.

    Args:
        book: State after today's sales (pools modified in place).
        price: (S,) quotes.
        t: Day index.
        traded: (N, S) securities sold today (modified: new picks added).
        rng: Generator of the day.

    Returns:
        ``(account, security, amount, kind)`` arrays of the orders
        (kind 1 = new position, 2 = top-up).
    """
    slot = t % (book.lag + 1)
    new_cash, new_cnt, rand_cash, under_cash = book.pools[:, :, slot].copy()
    book.pools[:, :, slot] = 0.0
    held = book.held()
    n_sec = held.shape[1]
    out: list[tuple[np.ndarray, ...]] = []
    a = np.nonzero(new_cnt > 0)[0]
    if a.size:
        c = new_cnt[a].astype(np.int64)
        keys = np.where(held[a] | traded[a], np.inf, rng.random((a.size, n_sec)))
        pick = np.argsort(keys, axis=1)[:, : c.max()]
        sec = pick[np.arange(c.max())[None, :] < c[:, None]]
        acc = np.repeat(a, c)
        traded[acc, sec] = True
        out.append((acc, sec, np.repeat(new_cash[a] / c, c), np.ones(acc.size, np.int8)))
    for cash, rule in ((rand_cash, "random"), (under_cash, "underweight")):
        b = np.nonzero(cash != 0.0)[0]
        if not b.size:
            continue
        cand = held[b] & ~traded[b]
        score = rng.random(cand.shape) if rule == "random" else book.pos_sh[b] * price
        target = np.where(cand, score, np.inf).argmin(1)
        none = ~cand.any(1)
        if none.any():
            keys = np.where(held[b[none]] | traded[b[none]], np.inf,
                            rng.random((none.sum(), n_sec)))
            target[none] = keys.argmin(1)
        out.append((b, target, cash[b], np.full(b.size, 2, np.int8)))
    if not out:
        return (np.zeros(0, np.int64),) * 2 + (np.zeros(0), np.zeros(0, np.int8))
    return tuple(np.concatenate(col) for col in zip(*out))


def _execute_buys(book: _Book, acc: np.ndarray, sec: np.ndarray, amount: np.ndarray,
                  kind: np.ndarray, price: np.ndarray, t: int, costs: CostConfig,
                  rec: _Recorder) -> tuple[np.ndarray, np.ndarray]:
    """Fill buy orders at ``mid*(1+s/2)`` and append a new lot to each.

    Orders on the same (account, security) are merged into one ticket. An
    order that cannot cover its commission is not sent and its cash stays
    idle. When a position already has ``max_lots`` lots the purchase is
    merged into the newest lot at the share-weighted price.

    Args:
        book: State (modified in place).
        acc, sec, amount, kind: Orders from :func:`_release_pools`.
        price: (S,) quotes.
        t: Day index.
        costs: Cost calibration.
        rec: Recorder (``buys`` table).

    Returns:
        ``(buy_value_mid, n_buy_tickets)`` per account, both (N,).
    """
    n, n_sec, max_lots = book.cash.size, price.size, book.lot_sh.shape[2]
    if not acc.size:
        return np.zeros(n), np.zeros(n, np.int64)
    uk, inv = np.unique(acc.astype(np.int64) * n_sec + sec, return_inverse=True)
    amt = np.bincount(inv, amount)
    is_new = np.bincount(inv, kind == 1) > 0
    acc, sec = np.divmod(uk, n_sec)
    ok = amt > costs.commission_usd
    book.cash += np.bincount(acc[~ok], amt[~ok], n)
    acc, sec, amt, is_new = acc[ok], sec[ok], amt[ok], is_new[ok]
    fill = price[sec] * (1.0 + costs.half_spread)
    sh = (amt - costs.commission_usd) / fill
    j = book.n_lots[acc, sec].astype(np.int64)
    merge, jj = j >= max_lots, np.minimum(j, max_lots - 1)
    old_sh, old_px = book.lot_sh[acc, sec, jj], book.lot_px[acc, sec, jj]
    new_sh = np.where(merge, old_sh + sh, sh)
    book.lot_px[acc, sec, jj] = np.where(merge, (old_sh * old_px + sh * fill) / new_sh, fill)
    book.lot_day[acc, sec, jj] = np.where(merge, book.lot_day[acc, sec, jj], t)
    book.lot_sh[acc, sec, jj] = new_sh
    book.n_lots[acc, sec] = np.minimum(j + 1, max_lots)
    book.pos_sh[acc, sec] += sh
    book.pos_cost[acc, sec] += sh * fill
    value = sh * price[sec]
    rec.add("buys", account=acc.astype(np.int32), day=np.full(acc.size, t, np.int16),
            security=sec.astype(np.int16), kind=np.where(is_new, 1, 2).astype(np.int8),
            value=value)
    return np.bincount(acc, value, n), np.bincount(acc, minlength=n)


@dataclass
class SimulationResult:
    """Everything one scenario run leaves behind (the "brokerage export").

    Daily arrays have shape (T, N); row t-1 refers to day t. Values are USD.

    Attributes:
        scenario, population: Inputs of the run.
        gross: Daily gross return at mid-quotes (all costs excluded).
        commission, spread: Daily commissions paid and spread cost
            (``half_spread * traded value at mid``).
        buy_value, sell_value: Daily traded value at mid.
        value_prev: Previous day's post-trade wealth (return denominator).
        cash_share: (N,) average share of wealth waiting in cash.
        tables: Record tables ``snapshots``, ``lots``, ``sales``, ``buys``,
            ``open_end``, each a dict of equal-length columns.
        runtime_s: Wall-clock seconds of the run.
    """

    scenario: ScenarioConfig
    population: Population
    gross: np.ndarray
    commission: np.ndarray
    spread: np.ndarray
    buy_value: np.ndarray
    sell_value: np.ndarray
    value_prev: np.ndarray
    cash_share: np.ndarray
    tables: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    runtime_s: float = 0.0

    @property
    def net(self) -> np.ndarray:
        """Daily net return: gross minus commissions and spread over wealth."""
        return self.gross - (self.commission + self.spread) / self.value_prev


def _day_rng(seed: np.random.SeedSequence, t: int) -> np.random.Generator:
    """Generator of day t: decisions of day t never depend on other days' draws."""
    return np.random.default_rng(child_seed(seed, 1, t))


def _record_open_end(book: _Book, price: np.ndarray, t: int, rec: _Recorder) -> None:
    """Record positions still open at the end (right-censored holding periods)."""
    acc, sec = np.nonzero(book.held())
    ls, dy = book.lot_sh[acc, sec], book.lot_day[acc, sec]
    hold = (ls * (t - dy)).sum(1) / ls.sum(1)
    rec.add("open_end", account=acc.astype(np.int32), security=sec.astype(np.int16),
            ret_vs_cost=price[sec] * book.pos_sh[acc, sec] / book.pos_cost[acc, sec] - 1.0,
            hold_days=hold)


def simulate_trading(market: Market, pop: Population, scn: ScenarioConfig,
                     beh: BehaviorConfig, costs: CostConfig,
                     seed: np.random.SeedSequence) -> SimulationResult:
    """Run the daily loop of one scenario on a fixed, pre-generated price path.

    Close of day t: mark at mid -> decide sales from today's and past quotes
    -> record the pre-trade snapshot of every selling account -> relieve lots
    and book proceeds -> release the pools due today into purchases -> mark
    again. Only ``market.prices[:t+1]`` is read (tested in Section 6).

    Args:
        market, pop, scn: Price universe, population and scenario.
        beh, costs: Behavioral and cost calibration.
        seed: Scenario seed; the population used sub-stream 0, days use 1.

    Returns:
        The :class:`SimulationResult`.
    """
    start = time.perf_counter()
    prices, n = market.prices, pop.n_agents
    n_days = prices.shape[0] - 1
    book = _initial_book(pop, prices[0], beh, scn.cash_lag_days, _day_rng(seed, 0))
    rec = _Recorder()
    names = ("gross", "commission", "spread", "buy_value", "sell_value", "value_prev")
    out = {k: np.zeros((n_days, n)) for k in names}
    cash_sum = np.zeros(n)
    v_prev = book.wealth(prices[0])
    for t in range(1, n_days + 1):
        rng, price = _day_rng(seed, t), prices[t]
        v_pre = book.wealth(price)
        trailing = prices[max(t - beh.mr_window, 0)]
        sold, kind = _decide_sales(book, price, trailing, pop, scn, beh, rng)
        traded = kind > 0
        rows = np.nonzero(traded.any(1))[0]
        sell_v, sell_n = np.zeros(n), np.zeros(n, np.int64)
        if rows.size:
            acc, sec, relief = _record_sale_day(book, rows, price, t, sold, kind, rec)
            sell_v, sell_n = _execute_sales(book, acc, sec, relief, sold, kind,
                                            price, t, costs, rec)
        orders = _release_pools(book, price, t, traded, rng)
        buy_v, buy_n = _execute_buys(book, *orders, price, t, costs, rec)
        v_post = book.wealth(price)
        out["gross"][t - 1] = v_pre / v_prev - 1.0
        out["commission"][t - 1] = costs.commission_usd * (sell_n + buy_n)
        out["spread"][t - 1] = costs.half_spread * (sell_v + buy_v)
        out["buy_value"][t - 1], out["sell_value"][t - 1] = buy_v, sell_v
        out["value_prev"][t - 1] = v_prev
        cash_sum += (v_post - (book.pos_sh * price).sum(1)) / v_post
        v_prev = v_post
    _record_open_end(book, prices[n_days], n_days, rec)
    tables = {k: rec.finish(k) for k in ("snapshots", "lots", "sales", "buys", "open_end")}
    return SimulationResult(scenario=scn, population=pop, cash_share=cash_sum / n_days,
                            tables=tables, runtime_s=time.perf_counter() - start, **out)
