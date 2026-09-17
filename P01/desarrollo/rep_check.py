import sys
import numpy as np, pandas as pd
sys.path.insert(0, r"C:\ifi\5\Comportamiento\proyecto_1")
import simlib_precios as sp, simlib_agentes as sa, simlib_estimadores as se

MASTER = np.random.SeedSequence(20260910)
MARKET = sp.simulate_market(sp.MarketConfig(), np.random.default_rng(sa.child_seed(MASTER, 0)))
BEH, COSTS = sa.BehaviorConfig(), sa.CostConfig()
rows = []
for r in range(6):
    for lab, corr, base in (("E9", 0.0, 700), ("E9m", 0.6, 720)):
        scn = sa.ScenarioConfig(f"{lab}#{r}", "rep", None, None, corr_delta_kappa=corr)
        seed = sa.child_seed(MASTER, base + r)
        sim = sa.simulate_trading(MARKET, sa.draw_population(scn, BEH, seed), scn, BEH, COSTS, seed)
        p = se.account_panel(sim, MARKET)
        w = se.account_weights(1000, 50, np.random.default_rng(r))
        a = se.turnover_slopes(p, w).set_index(["method", "outcome"])
        b = se.turnover_slopes(p, w, instrument="kappa").set_index(["method", "outcome"])
        rows.append({"pop": lab, "r": r, "ols": a.loc[("OLS", "gross"), "slope"], "ivh": a.loc[("IV", "gross"), "slope"],
                     "ivk": b.loc[("IV", "gross"), "slope"], "ivk_se": b.loc[("IV", "gross"), "se_hc1"],
                     "corr_dT": p["delta"].corr(p["turnover"])})
df = pd.DataFrame(rows)
print(df.round(3).to_string())
print(df.groupby("pop")[["ols", "ivh", "ivk", "corr_dT"]].agg(["mean", "std"]).round(3))
