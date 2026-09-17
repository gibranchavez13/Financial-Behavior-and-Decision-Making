import sys
import numpy as np, pandas as pd
sys.path.insert(0, r"C:\ifi\5\Comportamiento\proyecto_1")
import simlib_precios as sp, simlib_agentes as sa

MASTER = np.random.SeedSequence(20260910)
cfg = sp.MarketConfig()


def path_stats(m):
    """Realized cross-sectional reversal and the rebalancing premium of one path."""
    r = m.returns                                     # (T, S)
    ew_rebal = r.mean(1)                              # daily-rebalanced equal weight
    growth = np.vstack([np.ones(r.shape[1]), np.cumprod(1 + r, 0)])[:-1]   # buy-and-hold weights
    bh = (growth * r).sum(1) / growth.sum(1)
    half = r.shape[0] // 2
    x = np.log1p(r[:half]).sum(0) - m.betas * np.log1p(m.market_returns[:half]).sum()
    y = np.log1p(r[half:]).sum(0) - m.betas * np.log1p(m.market_returns[half:]).sum()
    # rolling version: 126-day past residual return vs next 126-day residual return
    lr = np.log1p(r) - np.outer(np.log1p(m.market_returns), m.betas)
    cs = np.vstack([np.zeros(r.shape[1]), np.cumsum(lr, 0)])
    cors = []
    for t in range(126, r.shape[0] - 126, 21):
        past, fut = cs[t] - cs[t - 126], cs[t + 126] - cs[t]
        cors.append(np.corrcoef(past, fut)[0, 1])
    return {"rebal_premium_pp": 25200 * (ew_rebal.mean() - bh.mean()),
            "corr_halves": np.corrcoef(x, y)[0, 1], "corr_rolling_126": np.mean(cors)}


base = sp.simulate_market(cfg, np.random.default_rng(sa.child_seed(MASTER, 0)))
print("base path:", {k: round(v, 3) for k, v in path_stats(base).items()})
mc = pd.read_csv(r"C:\ifi\5\Comportamiento\proyecto_1\resultados\monte_carlo_trayectorias.csv")
rows = []
for r_ in range(20):
    w = sp.simulate_market(cfg, np.random.default_rng(sa.child_seed(MASTER, 600 + r_)))
    rows.append({"mundo": r_, **path_stats(w)})
ps = pd.DataFrame(rows)
print(ps.describe().loc[["mean", "std", "min", "max"]].round(3))
for k in ("S1", "S3", "S5"):
    b = mc[mc["Esc."] == k].set_index("mundo")["b_bruta"]
    print(k, "corr(b_gross, rebal premium) =", round(np.corrcoef(b, ps["rebal_premium_pp"])[0, 1], 3),
          " corr(b_gross, rolling reversal) =", round(np.corrcoef(b, ps["corr_rolling_126"])[0, 1], 3))
