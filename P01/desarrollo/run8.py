import sys, time
import numpy as np, pandas as pd
sys.path.insert(0, r"C:\ifi\5\Comportamiento\proyecto_1")
import simlib_precios as sp, simlib_agentes as sa, simlib_estimadores as se
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)

master = np.random.SeedSequence(20260910)
mkt = sp.simulate_market(sp.MarketConfig(), np.random.default_rng(sa.child_seed(master, 0)))
beh, costs = sa.BehaviorConfig(), sa.CostConfig()
grid = [("S1", 0, 0, "none"), ("S2", .3, 0, "none"), ("S3", .8, 0, "none"), ("S4", 0, .3, "none"),
        ("S5", 0, .8, "none"), ("S6", .8, .8, "none"), ("S7", 0, 0, "rebalancing"), ("S8", 0, 0, "mean_reversion")]
rows, slopes, events = [], [], []
T0 = time.perf_counter()
for i, (k, d, kap, c) in enumerate(grid, start=1):
    scn = sa.ScenarioConfig(k, k, d, kap, c)
    seed = sa.child_seed(master, i)
    t0 = time.perf_counter()
    sim = sa.simulate_trading(mkt, sa.draw_population(scn, beh, seed), scn, beh, costs, seed)
    t1 = time.perf_counter()
    res = se.summarize_run(sim, mkt, 1000, np.random.default_rng(sa.child_seed(master, 300, i)))
    t2 = time.perf_counter()
    events.append(res["buys"])
    base = res["conventions"]["avg_count"]
    row = {"scn": k, "sim_s": t1 - t0, "est_s": t2 - t1, "PGR": base["pgr"], "PLR": base["plr"],
           "diff": base["diff"], "diff_se": base["diff_se"], "ratio": base["ratio"], "ratio_se": base["ratio_se"],
           "naive_se": res["naive"]["diff_se"], "se_ratio": base["diff_se"] / res["naive"]["diff_se"],
           "ties": base["ties"] / base["observations"], "turn": res["panel"]["turnover"].mean(),
           "cost": res["panel"]["costs"].mean()}
    for cv in ("avg_fraction", "fifo_count", "last_count", "lot_count"):
        row[cv] = res["conventions"][cv]["diff"]
    rows.append(row)
    s = res["slopes"].set_index(["method", "outcome"])
    slopes.append({"scn": k, **{f"{m}_{o}": s.loc[(m, o), "slope"] for m in ("OLS", "IV") for o in ("gross", "net", "costs", "gross_fill", "gross_log")},
                   "OLS_gross_se": s.loc[("OLS", "gross"), "se_boot"], "OLS_gross_hc1": s.loc[("OLS", "gross"), "se_hc1"],
                   "OLS_net_se": s.loc[("OLS", "net"), "se_boot"], "IV_gross_se": s.loc[("IV", "gross"), "se_boot"],
                   "F": s.loc[("IV", "gross"), "first_stage_F"]})
print(pd.DataFrame(rows).round(4).to_string())
print(pd.DataFrame(slopes).round(3).to_string())
for h in (20, 120):
    for adj in (False, True):
        print(se.leakage_test(events, mkt, h, adj))
print("total", time.perf_counter() - T0)
