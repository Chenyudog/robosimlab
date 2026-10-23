"""Matched h50, 5mm displacement diagnostic."""
import os
from run_baseline import Policy,ROOT
from run_paired_rollouts import run_pairs
if __name__ == '__main__':
    policy=Policy(ROOT/'models/pi0_libero_finetuned',vision_fp32=True)
    run_pairs(policy,ROOT/'outputs'/os.environ.get('VLA_PAIR_BASELINE','legacy_h50_spatial0_initial0_h50_seed7'),ROOT/'outputs'/os.environ.get('VLA_PAIR_OUTPUT','legacy_h50_paired_5mm'),max_steps=220,delta=.005,replan_steps=int(os.environ.get("VLA_PAIR_REPLAN","50")))
