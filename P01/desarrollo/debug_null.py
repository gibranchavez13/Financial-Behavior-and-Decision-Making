import sys, time
import numpy as np, pandas as pd
sys.path.insert(0, r"C:\ifi\5\Comportamiento\proyecto_1")
import simlib_precios as sp, simlib_agentes as sa, simlib_estimadores as se

master = np.random.SeedSequence(20260910)
beh, costs = sa.BehaviorConfig(), sa.CostConfig()
scn = sa.ScenarioConfig("S1", "null", 0.0, 0.0)

def slopes(mkt, seed):
    sim = sa.simulate_trading(mkt, sa.draw_population(scn, beh, seed), scn, beh, costs, seed)
    panel = se.account_panel(sim, mkt)
    w = se.account_weights(1000, 300, np.random.default_rng(1))
    s = se.turnover_slopes(panel, w).set_index(["method", "outcome"])
    # also a regression without the beta control and a reduced form on h0
    x = np.column_stack([np.ones(1000), panel["turnover"], panel["log_wealth"], panel["n_positions"]])
    nob = se.fit_linear(panel[["gross"]].to_numpy(), x)
    return {"ols": s.loc[("OLS", "gross"), "slope"], "ols_se": s.loc[("OLS", "gross"), "se_boot"],
            "iv": s.loc[("IV", "gross"), "slope"], "iv_se": s.loc[("IV", "gross"), "se_boot"],
            "ols_nobeta": nob["coef"][1, 0],
            "corr_turn_beta": np.corrcoef(panel["turnover"], panel["beta"])[0, 1]}

base_mkt = sp.simulate_market(sp.MarketConfig(), np.random.default_rng(sa.child_seed(master, 0)))
out = []
for r in range(12):   # same price path, new populations
    out.append({"design": "same path, new pop", **slopes(base_mkt, sa.child_seed(master, 900, r))})
for r in range(12):   # new price paths, same population seed
    mkt = sp.simulate_market(sp.MarketConfig(), np.random.default_rng(sa.child_seed(master, 901, r)))
    out.append({"design": "new path, same pop", **slopes(mkt, sa.child_seed(master, 1))})
for r in range(6):    # GBM paths
    mkt = sp.simulate_market(sp.MarketConfig(gbm=True), np.random.default_rng(sa.child_seed(master, 902, r)))
    out.append({"design": "GBM path", **slopes(mkt, sa.child_seed(master, 1))})
df = pd.DataFrame(out)
print(df.round(3).to_string())
g = df.groupby("design")
print(pd.concat([g.mean().add_suffix("_mean"), g[["ols", "iv"]].std().add_suffix("_sd")], axis=1).round(3).T)
