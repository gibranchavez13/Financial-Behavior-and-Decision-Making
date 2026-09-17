import sys, time, cProfile, pstats
import numpy as np
sys.path.insert(0, r"C:\ifi\5\Comportamiento\proyecto_1")
import simlib_precios as sp, simlib_agentes as sa, simlib_estimadores as se

MASTER = np.random.SeedSequence(20260910)
beh, costs = sa.BehaviorConfig(), sa.CostConfig()
scn = sa.ScenarioConfig("S3", "x", 0.8, 0.0)
world = sp.simulate_market(sp.MarketConfig(), np.random.default_rng(sa.child_seed(MASTER, 600)))
seed = sa.child_seed(MASTER, 500)
t0, c0 = time.perf_counter(), time.process_time()
pr = cProfile.Profile(); pr.enable()
sim = sa.simulate_trading(world, sa.draw_population(scn, beh, seed), scn, beh, costs, seed)
t1, c1 = time.perf_counter(), time.process_time()
res = se.summarize_run(sim, world, 200, np.random.default_rng(1))
pr.disable()
t2, c2 = time.perf_counter(), time.process_time()
print(f"sim wall {t1-t0:.1f}s cpu {c1-c0:.1f}s | est wall {t2-t1:.1f}s cpu {c2-c1:.1f}s")
pstats.Stats(pr).sort_stats("cumulative").print_stats(18)
