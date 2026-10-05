import sys, numpy as np
from strict_pipeline import (SimConfig, generate_dataset, split_time_series, fit_training_models,
                             validation_calibration, rule_B, rule_C_count, evaluate_rule)
for seed in [20260924, 3]:
    cfg=SimConfig(seed=seed); cfg.bootstrap_metrics=0; cfg.bootstrap_gpd=0
    X,y,_=generate_dataset(cfg); d,_=split_time_series(X,y)
    Xtr,ytr=d["train"]; Xv,yv=d["val"]; Xte,yte=d["test"]
    gpd,marg,cop,_,_=fit_training_models(Xtr,cfg)
    cal=validation_calibration(Xv,yv,cfg,gpd,marg,cop["theta"],cop["family"])
    B=rule_B(Xte,cal["B"]["thresholds"])
    Bq=rule_B(Xte, np.array(cal["B"]["thresholds"]))        # B at C22's threshold
    C22=rule_C_count(Xte,cal["C22"]["thresholds"])
    C22_atBq=rule_C_count(Xte, np.array(cal["B"]["thresholds"]))
    tol=cfg.early_tolerance_steps
    print("seed",seed,"B_p=%.5f C22_p=%.5f"%(cal["B"]["p"],cal["C22"]["p"]))
    print("  C22 & ~B steps:", int(np.sum(C22&~B)),
          "| C22_at_Bq & ~B steps:", int(np.sum(C22_atBq&~B)), "(should be 0)")
    a,Bq2=rule_B(Xte,np.array(cal["B"]["thresholds"])),np.array(cal["B"]["thresholds"])
    for tag,al in [("B",B),("C22 own q",C22),("C22 at B q",C22_atBq)]:
        m=evaluate_rule(al,yte,cfg.interval_sec,tol)
        print("   %-12s event_fnr=%.3f far=%.4f n_danger=%d matched=%d"%(tag,m["event_fnr"],m["far_per_hour"],m["n_danger_events"],m["n_matched"]))
