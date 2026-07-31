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
        --prime 1009 \
        [--kira-result-dir ~/projects/IBPswithAI/kira_tb105]

With --kira-result-dir, also parses Kira's truth reduction and reports
which model actions are consistent with the Kira result.
"""

import sys
import re
import gzip
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
    parser.add_argument('--kira-result-dir', default=None,
                        help='Path to a completed kira_tb105 directory; '
                             'parses results/TB/kira and tmp/TB/SYSTEM_TB_*.gz '
                             'to show truth reduction and Kira-equation matching.')
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
    # Identify the paper masters for this sector
    from generate_multisector_data import get_masters_for_sector, PRIME as _P
    paper_masters = set(get_masters_for_sector(sector_id))
    print(f"Masters for sector {sector_id}: {[list(m) for m in paper_masters]}")

    perfect_indices   = []  # all non-masters gone in one step
    reduces_indices   = []  # strictly lower max weight after step
    hits_master_indices = []  # master appears in result (multi-step)
    action_results = []

    for idx, (ibp_op, delta) in enumerate(valid_actions):
        seed = tuple(integral[i] + delta[i] for i in range(topology.n_indices))
        raw = get_raw_equation(ibp_t, li_t, ibp_op, seed)
        sol = solve_ibp_for(raw, integral)
        if sol is None:
            action_results.append(None)
            continue
        new_expr = apply_substitution(expr, integral, sol)
        non_masters = {k: v for k, v in new_expr.items() if v != 0 and not is_master(k)}
        has_paper_master = bool(k for k in new_expr if new_expr[k] != 0 and k in paper_masters)
        orig_w = weight(integral)

        if not non_masters:
            perfect_indices.append(idx)
            action_results.append(('PERFECT', new_expr))
        elif non_masters and max(weight(k) for k in non_masters) < orig_w:
            reduces_indices.append(idx)
            action_results.append(('REDUCES', new_expr, max(weight(k) for k in non_masters)))
        elif has_paper_master:
            hits_master_indices.append(idx)
            action_results.append(('HITS_MASTER', new_expr))
        else:
            action_results.append(('NO_PROGRESS', new_expr))

    all_useful = perfect_indices + reduces_indices + hits_master_indices
    print(f"Actions: perfect={len(perfect_indices)}  reduces_weight={len(reduces_indices)}  hits_master={len(hits_master_indices)}")
    if all_useful:
        for ci in all_useful[:10]:
            print(f"  action[{ci}] = (op={valid_actions[ci][0]}, delta={list(valid_actions[ci][1])})  -> {action_results[ci][0]}")

    # --- Kira truth (optional) ---
    kira_truth_str = None
    kira_sailir_matches = {}  # action_idx -> kira eq fingerprint
    if args.kira_result_dir:
        kira_truth_str, kira_sailir_matches = _load_kira(
            Path(args.kira_result_dir), integral, topology,
            valid_actions, ibp_t, li_t, args.prime)

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
        correct = "*** YES ***" if idx in all_useful else ""
        op, delta = valid_actions[idx]
        print(f"{rank+1:>5}  {probs[idx]:>8.4f}  {correct:>10}  {op}  {list(delta[:5])}")

    # Show rank of useful actions
    print(f"\nRank of useful actions (perfect / reduces / hits_master):")
    if all_useful:
        for ci in all_useful:
            rank = int(np.where(ranked == ci)[0][0]) + 1
            print(f"  action[{ci}]: rank {rank}/{len(valid_actions)}  prob={probs[ci]:.4f}  ({action_results[ci][0]})")
    else:
        print("  No single-step useful actions found.")
        print("  (This integral requires multi-step reduction.)")

    if kira_truth_str:
        print(f"\nKira truth: {kira_truth_str}")
        if kira_sailir_matches:
            print(f"SAILIR actions matching Kira IBP equations: {len(kira_sailir_matches)}")
            kira_ranks = sorted(kira_sailir_matches.keys(),
                                key=lambda ci: int(np.where(ranked == ci)[0][0]))
            for ci in kira_ranks[:10]:
                rank = int(np.where(ranked == ci)[0][0]) + 1
                tag = action_results[ci][0] if action_results[ci] else 'None'
                print(f"  action[{ci}]: rank {rank}/{len(valid_actions)}  "
                      f"prob={probs[ci]:.4f}  ({tag})  kira_eq={kira_sailir_matches[ci]}")
        else:
            print("  No SAILIR actions matched Kira IBP equations directly.")
            print("  (Kira likely used sector symmetries not in SAILIR's action space.)")


def _load_kira(kira_dir, integral, topology, valid_actions, ibp_t, li_t, prime):
    """Parse Kira result dir; return (truth_str, {action_idx: note}).

    truth_str: human-readable reduction formula for the target integral.
    action matches: currently empty — direct matching requires the full Kira
    id→integral map which is only available for the mandatory list entry.
    The Kira SYSTEM file shows the direct 1-step reduction uses a sector
    symmetry (momentum relabeling), not a pure IBP/LI — so no SAILIR action
    maps to it directly.
    """
    kira_result = kira_dir / 'results' / 'TB' / 'kira'
    if not kira_result.exists():
        print(f"  [kira] result file not found: {kira_result}")
        return None, {}

    text = kira_result.read_text()

    # Identify our integral's Kira ID from the id2int file
    integral_id = None
    id2int_path = kira_dir / 'results' / 'TB' / 'id2int'
    if id2int_path.exists():
        try:
            import yaml as _yaml
            rows = _yaml.safe_load(id2int_path.read_text()) or []
            for row in rows:
                if tuple(row[1:12]) == integral:
                    integral_id = row[0]
                    break
        except Exception:
            pass

    from generate_multisector_data import get_masters_for_sector as _get_masters
    int_sector = get_sector_id(integral)
    int_masters = list(_get_masters(int_sector))

    truth_str = None
    if integral_id is not None:
        # kira file format: "- Eq:\n  - [id,...,"1"]\n  - [1,...,"coeff"]\n"
        # master has id=1 in Kira convention
        blocks = re.split(r'- Eq:\s*\n', text)
        for block in blocks:
            entries = re.findall(r'\[(\d+),\d+,\d+,\d+,\d+,"([^"]+)"\]', block)
            if not entries:
                continue
            if int(entries[0][0]) == integral_id:
                for eid, ecoeff in entries:
                    if int(eid) == 1:
                        master_str = list(int_masters[0]) if int_masters else '?'
                        truth_str = (
                            f"TB{list(integral)} = -({ecoeff}) × TB{master_str}"
                        )
                        break
                break

    if truth_str is None:
        if integral_id is not None:
            m = re.search(
                rf'\[{integral_id},\d+,\d+,\d+,\d+,"1"\].*?\[1,\d+,\d+,\d+,\d+,"([^"]+)"\]',
                text, re.DOTALL)
            if m:
                truth_str = f"TB{list(integral)} = -({m.group(1)}) × master"
        if truth_str is None:
            truth_str = "(could not parse reduction formula from Kira output)"

    # Check Kira SYSTEM file for sector-symmetry vs IBP classification
    system_glob = list((kira_dir / 'tmp' / 'TB').glob('SYSTEM_TB_*.gz'))
    sector_sym_note = ""
    if system_glob and integral_id is not None:
        with gzip.open(system_glob[0], 'rt') as gf:
            sys_text = gf.read()
        # Look for blocks seeded by our integral_id
        seed_block_re = re.compile(
            rf'Eq\n{integral_id}\n(\d+)\n((?:.*\n)*?)(?=Eq|\Z)')
        n_direct_eqs = len(seed_block_re.findall(sys_text))
        # Check if any block with 2 terms is just a symmetry relation
        sym_blocks = re.findall(
            rf'Eq\n{integral_id}\n2\n(\d+) (-?\d+) (\d+) \d+ \d+ \d+\n'
            rf'(\d+) (-?\d+) (\d+) \d+ \d+ \d+\n',
            sys_text)
        if sym_blocks:
            sector_sym_note = (
                f"\nKira SYSTEM: found {len(sym_blocks)} 2-term equation(s) seeded by target "
                f"(likely sector symmetry, not in SAILIR action space)."
            )

    return (truth_str or "(no truth parsed)") + sector_sym_note, {}


if __name__ == '__main__':
    main()
