"""Predeclared h10 replication of the same two h50 snapshot experiments."""
from run_baseline import Policy,ROOT
from run_paired_rollouts import run_pairs
if __name__ == '__main__':
    policy=Policy(ROOT/'models/pi0_libero_finetuned',vision_fp32=True)
    for initial in (0,1):
        run_pairs(policy,ROOT/f'outputs/legacy_h50_spatial0_initial{initial}_h50_seed7',ROOT/f'outputs/paired_h10_5mm_initial{initial}',max_steps=220,delta=.005,replan_steps=10)
