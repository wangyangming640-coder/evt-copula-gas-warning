from pathlib import Path
from strict_pipeline import SimConfig, run_30_replicates

if __name__ == '__main__':
    out = Path(__file__).parent / 'results' / '30_replicates'
    cfg = SimConfig(seed=20260924)
    df, eq, tests = run_30_replicates(cfg, out, R=30)
    print('\n30组实验完成')
    print(df.groupby('rule')[['event_fnr','time_fnr','far_per_hour','mean_delay_min']].agg(['mean','std']))
    print('\nWilcoxon:')
    print(tests)
