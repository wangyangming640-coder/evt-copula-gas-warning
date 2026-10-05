from pathlib import Path
from strict_pipeline import SimConfig, run_negative_controls

if __name__ == '__main__':
    out = Path(__file__).parent / 'results' / 'negative_controls'
    df = run_negative_controls(SimConfig(seed=20260924), out)
    print(df)
