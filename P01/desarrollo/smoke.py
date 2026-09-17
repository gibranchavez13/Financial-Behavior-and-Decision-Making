import sys, time
import numpy as np
sys.path.insert(0, r"C:\ifi\5\Comportamiento\proyecto_1")
import simlib_precios as sp
import simlib_agentes as sa

master = np.random.SeedSequence(20260910)
mcfg = sp.MarketConfig()
mkt = sp.simulate_market(mcfg, np.random.default_rng(sa.child_seed(master, 0)))
print(sp.correlation_summary(mkt))
beh, costs = sa.BehaviorConfig(), sa.CostConfig()

def check(scn, key):
    seed = sa.child_seed(master, key)
    pop = sa.draw_population(scn, beh, seed)
    t0 = time.perf_counter()
    res = sa.simulate_trading(mkt, pop, scn, beh, costs, seed)
    dt = time.perf_counter() - t0
    # accounting identity: v_post(t) = v_prev(t+1)
    v_pre = res.value_prev * (1 + res.gross)
    v_post = v_pre - res.commission - res.spread
    err = np.abs(v_post[:-1] - res.value_prev[1:]).max()
    rel = err / res.value_prev.min()
    turnover = 0.5 * (res.buy_value + res.sell_value).sum(0) / res.value_prev.mean(0) * 252 / res.gross.shape[0]
    sn = res.tables["snapshots"]
    print(f"{scn.key:4s} {scn.confound:15s} lag={scn.cash_lag_days:2d} runtime={dt:5.1f}s "
          f"acct_err={rel:.2e} cash_share={res.cash_share.mean():.4f} "
          f"turnover_mean={turnover.mean():.2f} sd={turnover.std():.2f} "
          f"snap_rows={len(sn['account']) if sn else 0} multi_lot={np.mean(sn['n_lots']>1):.3f} "
          f"sales={len(res.tables['sales']['account'])} buys={len(res.tables['buys']['account'])}")
    return res

for scn, key in [
    (sa.ScenarioConfig("S1", "null", 0.0, 0.0), 1),
    (sa.ScenarioConfig("S5", "k.8", 0.0, 0.8), 5),
    (sa.ScenarioConfig("S7", "reb", 0.0, 0.0, "rebalancing"), 7),
    (sa.ScenarioConfig("S8", "mr", 0.0, 0.0, "mean_reversion"), 8),
    (sa.ScenarioConfig("S3", "d.8", 0.8, 0.0), 3),
    (sa.ScenarioConfig("S1L", "null lag", 0.0, 0.0, cash_lag_days=21), 1),
]:
    check(scn, key)
