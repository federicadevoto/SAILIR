#!/usr/bin/env python3
"""
Diagnostic: for a given integral, enumerate all valid IBP actions,
identify which ones reduce toward the master, then check how the
SAILIR model ranks those correct actions vs. the rest.

Usage (run from repo root):
    python <this_script> \
        --topology topology_input/hexabox \
        --integral="1,0,0,1,0,2,1,0,0,0,0" \
        --model-checkpoint checkpoints/hexabox_100k/best_model.pt \
        --prime 1009

The script prints each valid action with its model probability rank,
highlighting the ones that actually move toward the master.
"""

import sys
import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent  # repo root (reduction/ is one level down)
for p in ['reduction', 'data-gen', '']:
    sys.path.insert(0, str(REPO / p))

import torch
import numpy as np

from sailir.topology import Topology
from sailir import ibp_env
from sailir.ibp_env import (
    init_from_topology as env_init, set_prime, set_paper_masters_only,
    is_master, weight, IBPEnvironment,
)
from beam_search_utils import get_sector_mask, filter_to_sector
import beam_search_v7 as bs7
from sailir.classifier import IBPActionClassifier
from sailir.classifier_nosubs import IBPActionClassifierNoSubs
from generate_multisector_data import (
    init_from_topology as datagen_init,
    parse_templates, get_raw_equation, solve_ibp_for,
    enumerate_valid_actions, apply_substitution, get_sector_id,
    PRIME as DATAGEN_PRIME,
)
import generate_multisector_data as gmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topology', required=True)
    parser.add_argument('--integral', required=True)
    parser.add_argument('--model-checkpoint', required=True)
    parser.add_argument('--prime', type=int, default=1009)
    parser.add_argument('--no-paper-masters-only', action='store_true', default=True)
    args = parser.parse_args()

    gmd.PRIME = args.prime

    # --- topology + IBP env setup ---
    topology = Topology.from_dir(args.topology)
    env_init(topology)
    datagen_init(topology)
    set_prime(args.prime)
    set_paper_masters_only(not args.no_paper_masters_only)

    bs7._V7_REGISTRY = bs7.IntegralRegistry()
    bs7._V7_PACKED_RS_CACHE = {}
    bs7._PACKED_RS = False

    # --- model ---
    ck = torch.load(args.model_checkpoint, map_location='cpu', weights_only=False)
    model_variant = (ck.get('args') or {}).get('model_variant', 'full')
    ModelClass = IBPActionClassifierNoSubs if model_variant == 'nosubs' else IBPActionClassifier
    model = ModelClass(
        n_indices=topology.n_indices,
        n_denominators=topology.n_denominators,
        n_ibp_ops=topology.n_actions,
    )
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    print(f"Loaded {model_variant} model from {args.model_checkpoint}")

    # --- integral ---
    integral = tuple(int(x) for x in args.integral.strip("'\"").split(','))
    sector_id = get_sector_id(integral)
    sector_mask = tuple(get_sector_mask(integral))
    r, s = weight(integral)[:2]
    print(f"\nIntegral: {list(integral)}")
    print(f"Sector:   {sector_id}  mask={list(sector_mask)}")
    print(f"Weight:   r={r}  s={s}  t={sum(sector_mask)}")
    print(f"Is master: {is_master(integral)}")

    # --- IBP templates ---
    ibp_path = str(Path(args.topology) / 'IBP')
    li_path  = str(Path(args.topology) / 'LI')
    ibp_t = parse_templates(ibp_path)
    li_t  = parse_templates(li_path)
    n_ibp = len(ibp_t)
    num_ops = n_ibp + len(li_t)

    shifts = {}
    for op in range(n_ibp):
        if op in ibp_t:
            shifts[op] = [s for s, _ in ibp_t[op]]
    for li_idx in li_t:
        shifts[n_ibp + li_idx] = [s for s, _ in li_t[li_idx]]

    # --- enumerate valid actions ---
    expr = {integral: 1}
    subs = {}
    valid_actions = enumerate_valid_actions(
        integral, subs, ibp_t, li_t, shifts, sector_id, filter_higher=True)
    print(f"\nValid actions: {len(valid_actions)}")

    # --- for each valid action, evaluate the result ---
    # "correct" = applying the action and the result has lower max weight
    # i.e. all non-masters in the result have weight < (r, s) of the original
    correct_indices = []
    action_results = []
    for idx, (ibp_op, delta) in enumerate(valid_actions):
        seed = tuple(integral[i] + delta[i] for i in range(topology.n_indices))
        raw = get_raw_equation(ibp_t, li_t, ibp_op, seed)
        cached = raw  # no subs yet
        sol = solve_ibp_for(cached, integral)
        if sol is None:
            action_results.append(None)
            continue
        # Apply substitution to expr
        new_expr = apply_substitution(expr, integral, sol)
        # Find all non-masters in the result
        non_masters = {k: v for k, v in new_expr.items() if v != 0 and not is_master(k)}
        if not non_masters:
            # All terms are masters — perfect one-step reduction!
            correct_indices.append(idx)
            action_results.append(('PERFECT', new_expr))
        else:
            max_w = max(weight(k) for k in non_masters)
            orig_w = weight(integral)
            if max_w < orig_w:
                correct_indices.append(idx)
                action_results.append(('REDUCES', new_expr, max_w))
            else:
                action_results.append(('NO_PROGRESS', new_expr, max_w))

    print(f"Actions that reduce weight: {len(correct_indices)}")
    if correct_indices:
        for ci in correct_indices:
            print(f"  action[{ci}] = (op={valid_actions[ci][0]}, delta={list(valid_actions[ci][1])})  -> {action_results[ci][0]}")

    # --- model scoring ---
    print("\nRunning model forward pass...")
    batch_data = [(expr, subs, valid_actions, sector_mask, integral)]
    b = bs7.prepare_batched_input_v5_dummy(batch_data, device='cpu', max_actions=len(valid_actions)+10)
    with torch.no_grad():
        _, probs_t = model(
            b['expr_integrals'], b['expr_coeffs'], b['expr_mask'],
            b['sub_keys'], b['sub_repl_ints'], b['sub_repl_coeffs'],
            b['sub_repl_mask'], b['sub_mask'],
            b['action_ibp_ops'], b['action_deltas'], b['action_mask'],
            b['sector_mask'], b['target_integral'],
        )
    probs = probs_t[0, :len(valid_actions)].numpy()
    ranked = np.argsort(-probs)  # descending

    print(f"\nTop 10 actions by model probability:")
    print(f"{'Rank':>5}  {'Prob':>8}  {'Correct?':>10}  op  delta[:5]")
    print("-" * 60)
    for rank, idx in enumerate(ranked[:10]):
        correct = "*** YES ***" if idx in correct_indices else ""
        op, delta = valid_actions[idx]
        print(f"{rank+1:>5}  {probs[idx]:>8.4f}  {correct:>10}  {op}  {list(delta[:5])}")

    # Show rank of correct actions
    print(f"\nRank of correct actions:")
    for ci in correct_indices:
        rank = int(np.where(ranked == ci)[0][0]) + 1
        print(f"  action[{ci}]: rank {rank}/{len(valid_actions)}  prob={probs[ci]:.4f}  ({action_results[ci][0]})")

    if not correct_indices:
        print("  (no action found that reduces weight — unexpected!)")


if __name__ == '__main__':
    main()
