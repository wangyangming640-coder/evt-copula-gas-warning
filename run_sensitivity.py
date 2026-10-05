from pathlib import Path
from strict_pipeline import SimConfig, run_sensitivity

if __name__ == '__main__':
    out = Path(__file__).parent / 'results' / 'sensitivity'
    df = run_sensitivity(SimConfig(seed=20260924), out)
    print(df.groupby(['factor','rule'])[['event_fnr','far_per_hour']].mean())
