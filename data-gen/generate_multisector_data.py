#!/usr/bin/env python3
"""
Generate classifier training data for IBP reduction across ALL sectors.

TOPOLOGY-AGNOSTIC since C5d: this module reads dimensions, action templates,
and the master basis from a Topology (loaded with Topology.from_dir from a
topology_input/<family>/ directory). Run init_from_topology(t) before any
of the parsing / scrambling functions.

Each sample contains:
- sector_id: binary encoding of the sector being reduced (bits = denominator
             positions, in Kira convention)
- Current expression
- Previous substitutions
- Valid action space
- Chosen action (the label)

Output: JSONL file with one sample per reduction step.
"""

import re
import random
import json
import argparse
import sys
from pathlib import Path

# Make the SAILIR package importable when this script is run directly.
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
from sailir.topology import Topology

# Global prime - set by command line argument
PRIME = None

# ---------------------------------------------------------------------------
# Topology-driven globals (set by init_from_topology).
# ---------------------------------------------------------------------------
_TOPOLOGY: Topology = None
N_INDICES: int = 0
N_DENOMINATORS: int = 0
ISP_POSITIONS: tuple = ()
FAMILY_NAME: str = ""
KINEMATICS: dict = {}
PAPER_MASTERS: dict = {}                     # sector_id -> [master tuples]
SECTORS_WITH_PAPER_MASTERS: set = set()


def init_from_topology(topology: Topology) -> None:
    """Pull topology-dependent constants into module globals."""
    global _TOPOLOGY, N_INDICES, N_DENOMINATORS, ISP_POSITIONS, FAMILY_NAME
    global KINEMATICS, PAPER_MASTERS, SECTORS_WITH_PAPER_MASTERS
    _TOPOLOGY = topology
    N_INDICES = topology.n_indices
    N_DENOMINATORS = topology.n_denominators
    ISP_POSITIONS = tuple(topology.isp_positions)
    FAMILY_NAME = topology.family_name
    KINEMATICS = dict(topology.kinematics_values)
    PAPER_MASTERS = {s: [tuple(m) for m in ms]
                     for s, ms in topology.masters_by_sector.items()}
    SECTORS_WITH_PAPER_MASTERS = set(PAPER_MASTERS.keys())


def get_sector_id(integral):
    """Compute sector ID from integral indices.

    Sector ID is the bitmask of denominator positions (those NOT in
    ISP_POSITIONS) where integral[i] >= 1. Matches Kira's convention.
    """
    sector_id = 0
    for i in range(N_INDICES):
        if i in ISP_POSITIONS:
            continue
        if integral[i] >= 1:
            sector_id += (1 << i)
    return sector_id


def get_sector_mask(sector_id):
    """Tuple of length N_DENOMINATORS: 1 at each denom slot present in sector_id.

    Walks denominator positions in order (skipping ISP slots) so the returned
    tuple is indexed 0..N_DENOMINATORS-1 (same convention as the old trianglebox
    6-bit mask).
    """
    out = []
    for i in range(N_INDICES):
        if i in ISP_POSITIONS:
            continue
        out.append((sector_id >> i) & 1)
    return tuple(out)


def get_corner_integral(sector_id):
    """Corner integral for a sector: 1 at each present denom, 0 elsewhere."""
    return tuple(1 if (sector_id >> i) & 1 and i not in ISP_POSITIONS else 0
                  for i in range(N_INDICES))


def get_masters_for_sector(sector_id):
    """
    Get list of master integrals for a sector.
    Uses paper masters if available, otherwise corner integral.
    """
    if sector_id in PAPER_MASTERS:
        return [tuple(m) for m in PAPER_MASTERS[sector_id]]
    else:
        return [get_corner_integral(sector_id)]


def is_in_sector(integral, sector_id):
    """Is `integral` in `sector_id` or a subsector?

    True iff every denominator position required by sector_id has integral >= 1.
    Assumes denominator positions are 0..N_DENOMINATORS-1 (ISPs come last).
    """
    mask = get_sector_mask(sector_id)
    for i in range(N_DENOMINATORS):
        if mask[i] and integral[i] < 1:
            return False
    return True


def is_higher_sector(integral, sector_id):
    """Is `integral` in a strictly higher sector than sector_id?

    Higher = has all required denoms AND at least one extra denom (or any ISP > 0).
    """
    if not is_in_sector(integral, sector_id):
        return False
    mask = get_sector_mask(sector_id)
    for i in range(N_DENOMINATORS):
        if not mask[i] and integral[i] >= 1:
            return True
    # Any ISP with positive index counts as higher.
    if any(integral[p] >= 1 for p in ISP_POSITIONS):
        return True
    return False


def is_master_for_sector(integral, sector_id):
    """Check if integral is a master for the given sector."""
    masters = get_masters_for_sector(sector_id)
    return tuple(integral) in masters


def weight(integral):
    """Compute weight for ordering integrals (higher weight = reduce first)."""
    return (
        sum(max(0, x) for x in integral),
        -sum(min(0, x) for x in integral),
        tuple(abs(x) for x in integral)
    )


def filter_sector_only(expr, sector_id):
    """Filter to integrals in the given sector, excluding higher sectors."""
    return {k: v for k, v in expr.items()
            if is_in_sector(k, sector_id) and not is_higher_sector(k, sector_id)}


# =============================================================================
# Modular arithmetic
# =============================================================================

def mod_inverse(a, p=None):
    if p is None:
        p = PRIME
    return pow(a % p, p - 2, p)


def solve_ibp_for(ibp, integral):
    if integral not in ibp or ibp[integral] == 0:
        return None
    neg_inv = (PRIME - mod_inverse(ibp[integral])) % PRIME
    return {k: (neg_inv * v) % PRIME for k, v in ibp.items() if k != integral}


def apply_substitution(expr, sub_int, sol):
    if sub_int not in expr:
        return expr
    coeff = expr[sub_int]
    new_expr = {k: v for k, v in expr.items() if k != sub_int}
    for integral, sub_coeff in sol.items():
        new_coeff = (coeff * sub_coeff) % PRIME
        if integral in new_expr:
            new_expr[integral] = (new_expr[integral] + new_coeff) % PRIME
        else:
            new_expr[integral] = new_coeff
    return {k: v for k, v in new_expr.items() if v != 0}


def apply_all_substitutions(expr, subs):
    result = dict(expr)
    for sub_int, sol in subs.items():
        result = apply_substitution(result, sub_int, sol)
    return result


# =============================================================================
# IBP/LI template handling
# =============================================================================

def parse_templates(path):
    """Parse IBP or LI file. Family name comes from the configured topology."""
    if not FAMILY_NAME:
        raise RuntimeError("init_from_topology() must be called before parse_templates")
    pat = re.compile(rf'{re.escape(FAMILY_NAME)}\[([^\]]+)\]\*\(([^)]+)\)')
    templates = {}
    current_idx, terms = None, []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_idx is not None and terms:
                    templates[current_idx] = terms
                current_idx, terms = None, []
                continue
            if '\u2192' in line:  # original "→" arrow
                line = line.split('\u2192', 1)[1]
            match = pat.match(line)
            if match:
                shift = tuple(int(x.strip()) for x in match.group(1).split(','))
                if current_idx is None:
                    current_idx = len(templates)
                terms.append((shift, match.group(2)))
    if current_idx is not None and terms:
        templates[current_idx] = terms
    return templates


def evaluate_coefficient(coeff_str, seed):
    """Eval coefficient at finite-field point. Indices a0..a{N-1} from `seed`,
    plus topology kinematics (d + invariants) from KINEMATICS."""
    ns = {f'a{i}': seed[i] for i in range(N_INDICES)}
    ns.update(KINEMATICS)
    try:
        return eval(coeff_str.replace('^', '**'), {"__builtins__": {}}, ns) % PRIME
    except Exception:
        return 0


def get_raw_equation(ibp_t, li_t, ibp_op, seed):
    n_ibp = len(ibp_t)
    if ibp_op >= n_ibp:
        template = li_t.get(ibp_op - n_ibp, [])
    else:
        template = ibp_t.get(ibp_op, [])
    eq = {}
    for shift, coeff_str in template:
        coeff = evaluate_coefficient(coeff_str, seed)
        if coeff:
            eq[tuple(seed[i] + shift[i] for i in range(N_INDICES))] = coeff
    return eq


# =============================================================================
# Scrambling (reverse reduction)
# =============================================================================

def action_introduces_higher_sector(cached_eq, sector_id):
    """Check if an IBP equation introduces higher sector integrals."""
    for integral in cached_eq.keys():
        if is_in_sector(integral, sector_id) and is_higher_sector(integral, sector_id):
            return True
    return False


def get_integral_props(integral):
    """Get the set of propagator indices where integral has power >= 1."""
    return {i for i in range(N_DENOMINATORS) if integral[i] >= 1}


def is_lateral_sector(integral, sector_id):
    """Check if integral is in a lateral sector to sector_id.

    Lateral = integral's propagator set is neither a subset nor a superset
    of sector_id's propagator set.

    Example: sector 53 has props {0,2,4,5}
             sector 54 has props {1,2,4,5} - lateral (has 1, missing 0)
             sector 52 has props {2,4,5} - NOT lateral (subset)
             sector 55 has props {0,1,2,4,5} - NOT lateral (superset)
    """
    int_props = get_integral_props(integral)
    target_props = {i for i in range(N_DENOMINATORS) if (sector_id >> i) & 1}
    is_subset = int_props <= target_props
    is_superset = int_props >= target_props
    return not is_subset and not is_superset


def action_introduces_outside_sector(cached_eq, sector_id):
    """Check if an IBP equation introduces integrals outside the target sector's cone.

    An integral is "outside" if it's in a higher sector or a lateral sector.
    Subsectors are OK (their propagator set is a subset of target's).

    Returns True if any integral in cached_eq is:
    - In a higher sector (supersector of target) - has extra propagators
    - In a lateral sector - has some extra propagators AND missing some required ones
    """
    for integral in cached_eq.keys():
        if is_higher_sector(integral, sector_id):
            return True
        if is_lateral_sector(integral, sector_id):
            return True
    return False


def find_candidates(ibp_t, li_t, expr, num_ops, sector_id):
    """Find candidate (ibp_op, seed) pairs for scrambling."""
    candidates = []
    for integral in expr:
        if is_in_sector(integral, sector_id) and not is_higher_sector(integral, sector_id):
            for ibp_op in range(num_ops):
                raw = get_raw_equation(ibp_t, li_t, ibp_op, integral)
                if raw and any(k in expr and is_in_sector(k, sector_id) for k in raw):
                    candidates.append((ibp_op, integral))
    return candidates


def _is_pure_dot(k):
    """True if integral has no negative indices (no ISP numerators) and at least one index > 1."""
    return all(i >= 0 for i in k) and any(i > 1 for i in k)


def scramble(start, ibp_t, li_t, num_ops, n_steps, sector_id,
             filter_lateral=False, bias_low_s_elim=False, bias_dots_elim=False):
    """Scramble expression using only actions that don't introduce higher/lateral sectors.

    bias_low_s_elim: if True, when choosing which integral to eliminate from the
        IBP equation, prefer integrals with LOW s (few ISP numerators). The elim
        is replaced by the rest of the equation, so removing a low-s integral
        tends to LEAVE high-s ones in the expression — pushing scramble outputs
        toward the deep-ISP regime that the baseline underrepresents.

    bias_dots_elim: if True, prefer eliminating non-dot integrals from the IBP
        equation. A "pure dot" (all non-negative indices, at least one > 1) is
        10x less likely to be chosen as elim than a non-dot, so dots accumulate
        in the expression and appear more often as targets in the training data.
    """
    expr = dict(start)
    used_ibps = []

    for step in range(n_steps):
        candidates = find_candidates(ibp_t, li_t, expr, num_ops, sector_id)
        if not candidates:
            break

        random.shuffle(candidates)
        found_good_candidate = False

        for ibp_op, seed in candidates:
            raw = get_raw_equation(ibp_t, li_t, ibp_op, seed)

            # Check if this IBP introduces higher sector (and optionally lateral sectors)
            if filter_lateral:
                if action_introduces_outside_sector(raw, sector_id):
                    continue
            else:
                if action_introduces_higher_sector(raw, sector_id):
                    continue

            top_only = [k for k in raw
                       if is_in_sector(k, sector_id) and not is_higher_sector(k, sector_id)]
            if not top_only:
                continue

            if (bias_low_s_elim or bias_dots_elim) and len(top_only) > 1:
                # Weights are multiplicative so both flags can be active together.
                # bias_low_s_elim: 1/(1+s) → low-s integrals easily eliminated, keeping high-s in expr
                # bias_dots_elim:  0.1 for pure-dots → dots rarely eliminated, keeping them in expr
                w = []
                for k in top_only:
                    wi = 1.0
                    if bias_low_s_elim:
                        wi *= 1.0 / (1 + weight(k)[1])
                    if bias_dots_elim:
                        wi *= 0.1 if _is_pure_dot(k) else 1.0
                    w.append(wi)
                elim = random.choices(top_only, weights=w)[0]
            else:
                elim = random.choice(top_only)
            sol = solve_ibp_for(raw, elim)
            if sol is None:
                continue

            if elim in expr:
                expr = apply_substitution(expr, elim, sol)
            else:
                for k, v in raw.items():
                    expr[k] = (expr.get(k, 0) + v) % PRIME
                expr = {k: v for k, v in expr.items() if v != 0}

            used_ibps.append((ibp_op, seed))
            found_good_candidate = True
            break

        if not found_good_candidate:
            break

    return expr, used_ibps


# =============================================================================
# Action enumeration
# =============================================================================

def enumerate_valid_actions(target, subs, ibp_t, li_t, shifts, sector_id,
                           filter_higher=True, filter_lateral=False):
    """
    Enumerate all valid (ibp_op, delta) pairs that can eliminate target.
    Returns list of (ibp_op, delta) where delta = seed - target.

    Args:
        filter_higher: Filter out actions introducing higher sector integrals
        filter_lateral: Also filter out actions introducing lateral sector integrals
    """
    valid = []
    seen = set()

    # Direct actions: target directly in raw_ibp
    for ibp_op, shift_list in shifts.items():
        for shift in shift_list:
            seed = tuple(target[i] - shift[i] for i in range(N_INDICES))
            raw = get_raw_equation(ibp_t, li_t, ibp_op, seed)
            if target not in raw or raw[target] == 0:
                continue
            cached = apply_all_substitutions(raw, subs)
            if target not in cached or cached[target] == 0:
                continue
            # Filter based on options
            if filter_higher:
                if filter_lateral:
                    if action_introduces_outside_sector(cached, sector_id):
                        continue
                else:
                    if action_introduces_higher_sector(cached, sector_id):
                        continue
            delta = tuple(seed[i] - target[i] for i in range(N_INDICES))
            if (ibp_op, delta) not in seen:
                seen.add((ibp_op, delta))
                valid.append((ibp_op, delta))

    # Indirect actions: target via substitution chain
    for sub_int in subs:
        for ibp_op, shift_list in shifts.items():
            for shift in shift_list:
                seed = tuple(sub_int[i] - shift[i] for i in range(N_INDICES))
                raw = get_raw_equation(ibp_t, li_t, ibp_op, seed)
                if sub_int not in raw or raw[sub_int] == 0:
                    continue
                if target in raw and raw[target] != 0:
                    continue
                cached = apply_all_substitutions(raw, subs)
                if target not in cached or cached[target] == 0:
                    continue
                # Filter based on options
                if filter_higher:
                    if filter_lateral:
                        if action_introduces_outside_sector(cached, sector_id):
                            continue
                    else:
                        if action_introduces_higher_sector(cached, sector_id):
                            continue
                delta = tuple(seed[i] - target[i] for i in range(N_INDICES))
                if (ibp_op, delta) not in seen:
                    seen.add((ibp_op, delta))
                    valid.append((ibp_op, delta))

    return valid


# =============================================================================
# JSON serialization
# =============================================================================

def expr_to_json(expr):
    return [[list(k), v] for k, v in sorted(expr.items())]


def subs_to_json(subs):
    return [[list(k), [[list(ki), vi] for ki, vi in sorted(v.items())]]
            for k, v in sorted(subs.items())]


# =============================================================================
# Main
# =============================================================================

def get_all_valid_sectors_legacy_trianglebox():
    """Trianglebox-specific: 63 non-trivial sectors (1..63)."""
    return list(range(1, 64))


def get_all_valid_sectors():
    """All candidate corner-sectors for the configured topology.

    Returns every non-empty bitmask over the topology's N_DENOMINATORS
    propagator positions, i.e., range(1, 2**N_DENOMINATORS).

    For trianglebox this is 1..63 (=63 sectors), for pentagon-box 1..255
    (=255 sectors). Note that SAILIR's data generator intentionally
    iterates over ALL such corners, including ones Kira would flag as
    trivial; the trivial ones just produce empty/zero trajectories.
    """
    return list(range(1, 1 << N_DENOMINATORS))


def main():
    global PRIME

    parser = argparse.ArgumentParser(
        description='Generate multi-sector IBP reduction training data')
    parser.add_argument('--topology', type=str, required=True,
                        help='Path to topology_input/<family>/ directory '
                             '(e.g. topology_input/trianglebox)')
    parser.add_argument('--n_scrambles', type=int, default=1000,
                        help='Number of scramble trajectories to generate')
    parser.add_argument('--min_steps', type=int, default=5,
                        help='Minimum scramble steps')
    parser.add_argument('--max_steps', type=int, default=20,
                        help='Maximum scramble steps')
    parser.add_argument('--output', type=str,
                        default='data/multisector_training_data.jsonl',
                        help='Output file path')
    parser.add_argument('--start_seed', type=int, default=0,
                        help='Starting random seed')
    parser.add_argument('--prime', type=int, default=1009,
                        help='Prime for modular arithmetic')
    parser.add_argument('--ibp_path', type=str, default=None,
                        help='Path to IBP templates (defaults to <topology>/IBP)')
    parser.add_argument('--li_path', type=str, default=None,
                        help='Path to LI templates (defaults to <topology>/LI)')
    parser.add_argument('--filter_lateral', action='store_true',
                        help='Also filter out actions that introduce lateral sector integrals')
    parser.add_argument('--bias-low-s-elim', action='store_true',
                        help='Bias scramble to prefer eliminating low-s integrals '
                             '(so substituted-in terms are high-s). Pushes the '
                             'output (r,s) distribution toward deep ISPs.')
    parser.add_argument('--bias-dots-elim', action='store_true',
                        help='Bias scramble to prefer eliminating non-dot integrals. '
                             'A "pure dot" (all non-negative indices, at least one > 1) '
                             'is 10x less likely to be chosen as elim, so dots accumulate '
                             'in the scrambled expression and appear as training targets.')
    parser.add_argument('--restrict-sectors', type=str, default=None,
                        help='Comma-separated sector_ids to restrict scrambling to. '
                             'If omitted, all valid sectors are used.')
    args = parser.parse_args()

    PRIME = args.prime

    # Configure topology globals (N_INDICES, ISP_POSITIONS, KINEMATICS,
    # PAPER_MASTERS, ...) before any parsing happens.
    topology = Topology.from_dir(args.topology)
    init_from_topology(topology)

    if args.ibp_path is None:
        args.ibp_path = str(Path(args.topology) / 'IBP')
    if args.li_path is None:
        args.li_path = str(Path(args.topology) / 'LI')

    print(f"=" * 70, flush=True)
    print(f"Multi-Sector IBP Training Data Generator", flush=True)
    print(f"=" * 70, flush=True)
    print(f"Topology: {topology.name} (family={topology.family_name}, "
          f"n_indices={N_INDICES}, n_denominators={N_DENOMINATORS}, "
          f"ISPs={ISP_POSITIONS})", flush=True)
    print(f"PRIME = {PRIME}", flush=True)
    print(f"Scrambles: {args.n_scrambles}", flush=True)
    print(f"Steps: {args.min_steps}-{args.max_steps}", flush=True)
    print(f"Output: {args.output}", flush=True)
    print(f"Filter lateral sectors: {args.filter_lateral}", flush=True)

    if args.restrict_sectors:
        sector_list = [int(s) for s in args.restrict_sectors.split(',')]
        print(f"Restricted to {len(sector_list)} sectors: {sector_list}", flush=True)
    else:
        sector_list = get_all_valid_sectors()
        print(f"Using all {len(sector_list)} sectors "
              f"(1..{(1 << N_DENOMINATORS) - 1})", flush=True)
    print(f"bias_low_s_elim: {args.bias_low_s_elim}", flush=True)
    print(f"bias_dots_elim: {args.bias_dots_elim}", flush=True)

    # Load IBP/LI templates
    ibp_t = parse_templates(args.ibp_path)
    li_t = parse_templates(args.li_path)
    n_ibp = len(ibp_t)
    n_li = len(li_t)
    num_ops = n_ibp + n_li
    print(f"Loaded {n_ibp} IBP templates, {n_li} LI templates "
          f"(action space size {num_ops})", flush=True)

    # Build shifts lookup. Action indices 0..n_ibp-1 are IBPs, n_ibp..n_ibp+n_li-1 are LIs.
    shifts = {}
    for ibp_op in range(n_ibp):
        if ibp_op in ibp_t:
            shifts[ibp_op] = [s for s, _ in ibp_t[ibp_op]]
    for li_idx in li_t:
        shifts[n_ibp + li_idx] = [s for s, _ in li_t[li_idx]]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    total_samples = 0
    successful_scrambles = 0
    failed_scrambles = 0
    skipped_vanishing = 0  # Sectors with vanishing corners
    sector_counts = {}  # Track samples per sector

    with open(args.output, 'w') as f:
        for scramble_idx in range(args.n_scrambles):
            seed_value = args.start_seed + scramble_idx * 1000
            random.seed(seed_value)

            # Sample a sector
            sector_id = random.choice(sector_list)
            masters = get_masters_for_sector(sector_id)

            # Create random linear combination of masters
            start = {}
            master_coeffs = []
            for m in masters:
                c = random.randint(1, PRIME - 1)
                start[m] = c
                master_coeffs.append(c)

            # Scramble
            n_steps = random.randint(args.min_steps, args.max_steps)
            scrambled, used_ibps = scramble(start, ibp_t, li_t, num_ops, n_steps, sector_id,
                                           filter_lateral=args.filter_lateral,
                                           bias_low_s_elim=args.bias_low_s_elim,
                                           bias_dots_elim=args.bias_dots_elim)

            if not used_ibps:
                # Check if this is a vanishing corner (expression became empty or trivial)
                # This happens when the corner integral is zero by IBP identities
                skipped_vanishing += 1
                print(f"SKIPPED_VANISHING: scramble={scramble_idx}, sector={sector_id}, "
                      f"masters={masters}, reason=no_valid_scramble_actions", flush=True)
                continue

            # Unscramble and collect training samples
            expr = dict(scrambled)
            subs = {}
            used_set = set()
            scramble_samples = []
            success = True
            failure_reason = None

            for iteration in range(500):
                sector_only = filter_sector_only(expr, sector_id)
                non_masters = {k: v for k, v in sector_only.items()
                              if not is_master_for_sector(k, sector_id)}

                if not non_masters:
                    # Success - verify coefficients
                    final = {k: v for k, v in sector_only.items()
                            if is_master_for_sector(k, sector_id)}
                    expected_coeffs = {m: c for m, c in zip(masters, master_coeffs)}
                    if final != expected_coeffs:
                        success = False
                        failure_reason = f"coeff_mismatch: final={final}, expected={expected_coeffs}"
                    break

                target = sorted(non_masters.keys(),
                    key=lambda x: (-weight(x)[0], -weight(x)[1], weight(x)[2]))[0]

                # Find the used_ibps action that works
                found_action = None
                for idx, (ibp_op, seed) in enumerate(used_ibps):
                    if idx in used_set:
                        continue
                    raw = get_raw_equation(ibp_t, li_t, ibp_op, seed)
                    cached = apply_all_substitutions(raw, subs)
                    if target in cached and cached[target] != 0:
                        found_action = (idx, ibp_op, seed)
                        break

                if found_action is None:
                    success = False
                    failure_reason = f"no_action_found: iteration={iteration}, target={target}"
                    break

                idx, chosen_ibp_op, chosen_seed = found_action
                chosen_delta = tuple(chosen_seed[i] - target[i] for i in range(N_INDICES))

                # Enumerate valid actions
                valid_actions = enumerate_valid_actions(
                    target, subs, ibp_t, li_t, shifts, sector_id,
                    filter_lateral=args.filter_lateral)

                # Verify chosen action is in valid actions
                if (chosen_ibp_op, chosen_delta) not in valid_actions:
                    success = False
                    failure_reason = f"action_not_valid: iteration={iteration}, action=({chosen_ibp_op}, {chosen_delta})"
                    break

                chosen_idx = valid_actions.index((chosen_ibp_op, chosen_delta))

                # Create training sample with sector_id
                sample = {
                    'scramble_id': scramble_idx,
                    'sector_id': sector_id,  # NEW: include sector for model conditioning
                    'sector_mask': list(get_sector_mask(sector_id)),  # NEW: 6-bit mask
                    'step': iteration,
                    'target': list(target),
                    'target_weight': weight(target)[:2],
                    'expr': expr_to_json(filter_sector_only(expr, sector_id)),
                    'subs': subs_to_json(subs),
                    'valid_actions': [[ibp_op, list(delta)] for ibp_op, delta in valid_actions],
                    'num_valid_actions': len(valid_actions),
                    'chosen_action': [chosen_ibp_op, list(chosen_delta)],
                    'chosen_action_idx': chosen_idx
                }
                scramble_samples.append(sample)

                # Apply the action
                raw = get_raw_equation(ibp_t, li_t, chosen_ibp_op, chosen_seed)
                cached = apply_all_substitutions(raw, subs)
                sol = solve_ibp_for(cached, target)

                used_set.add(idx)
                subs[target] = sol
                expr = apply_substitution(expr, target, sol)
            else:
                success = False
                failure_reason = "max_iterations_reached"

            if success:
                for sample in scramble_samples:
                    f.write(json.dumps(sample) + '\n')
                total_samples += len(scramble_samples)
                successful_scrambles += 1
                sector_counts[sector_id] = sector_counts.get(sector_id, 0) + len(scramble_samples)
            else:
                # Coefficient mismatch typically indicates vanishing corner
                if failure_reason and failure_reason.startswith("coeff_mismatch"):
                    skipped_vanishing += 1
                    print(f"SKIPPED_VANISHING: scramble={scramble_idx}, sector={sector_id}, "
                          f"masters={masters}, reason={failure_reason}", flush=True)
                else:
                    failed_scrambles += 1
                    print(f"FAILED: scramble={scramble_idx}, sector={sector_id}, "
                          f"masters={masters}, reason={failure_reason}", flush=True)

            if (scramble_idx + 1) % 100 == 0:
                print(f"Progress: {scramble_idx + 1}/{args.n_scrambles}, "
                      f"samples={total_samples}, success={successful_scrambles}, "
                      f"skipped_vanishing={skipped_vanishing}, fail={failed_scrambles}", flush=True)

    print(f"\n{'=' * 70}", flush=True)
    print(f"Done! {total_samples} training samples from {successful_scrambles} scrambles", flush=True)
    print(f"Skipped (vanishing corners): {skipped_vanishing}", flush=True)
    print(f"Failed scrambles: {failed_scrambles}", flush=True)
    print(f"Output: {args.output}", flush=True)
    print(f"\nSamples per sector:", flush=True)
    for sid in sorted(sector_counts.keys()):
        print(f"  Sector {sid}: {sector_counts[sid]} samples", flush=True)


if __name__ == "__main__":
    main()
