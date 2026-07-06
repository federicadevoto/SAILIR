#!/usr/bin/env python3
"""onestep_worker_v7.py
=================================================================
Hierarchical-reduction worker that runs `beam_search_v7.beam_search_v5`
IN-PROCESS — exactly the way onestep_worker_v6.py runs v6, and exactly the way
the verified probes ran `python beam_search_v7.py --n-threads 8 --n-workers 8`.

WHY IN-PROCESS (and not a subprocess wrapper):
  A single process means beam_search_v7's import-time _cap_incidental_threads()
  runs in THIS process and confines it + its 8 forked enumerate workers to 8
  cores — byte-for-byte the CPU behavior the probes verified (8/8, zero holds).
  There is no pickle round-trip (so no `ModuleNotFoundError: sailir`), no stdout
  buffering (so the .out shows live progress and survives a kill), and the
  orchestrator-format result.pkl is written directly (so reaping just works).

v7 SETTINGS reproduced (== the success-only probe recipe):
  ENV (set BELOW, before importing beam_search_v7, because _SUCCESS_TOTAL /
       _BEAM_TOTAL / _ACTION_SELECT are read at module import):
       SAILIR_SUCCESS_TOTAL=1  SAILIR_ACTION_SELECT=maxweight
       SAILIR_STRIP_RAWS=1  SAILIR_PACKED_RS=1  SAILIR_END_OF_STEP_TRIM=1
       SAILIR_TABU_CAP=0  MALLOC_MMAP_THRESHOLD_=67108864  PYTHONUNBUFFERED=1
       (SAILIR_BEAM_TOTAL deliberately UNSET -> (r,s) beam + (r,s) maxweight)
  ARGV injection: --n-threads 8 --n-workers 8 prepended so the import-time
       _cap_incidental_threads() (which peeks sys.argv) applies the 8/8 thread
       caps + MKL GNU layer + 8-core affinity pin, identical to the probe.
  beam_search_v5 call: tabu=True, use_exprkeyed=False, iraws_keep_first=50,
       lazy_rs=True, max_actions=900, beam_sort='weight', model_batch_chunk=8,
       n_workers=8 (the fork pool).

CLI mirrors onestep_worker_v6.py so hierarchical_reduction.py dispatches to it
unchanged. Output keys match the orchestrator contract:
  success, original_integral, final_expr, path, restart_offsets, steps, time,
  prime, peak_memory_kb.
"""
import os
import sys

# ── v7 toggles: MUST be set before importing beam_search_v7 (module-level
#    _SUCCESS_TOTAL/_BEAM_TOTAL/_ACTION_SELECT read os.environ at import). ──
os.environ.setdefault('PYTHONUNBUFFERED', '1')
os.environ.setdefault('MALLOC_MMAP_THRESHOLD_', '67108864')
os.environ['SAILIR_END_OF_STEP_TRIM'] = '1'
os.environ['SAILIR_TABU_CAP'] = '0'
os.environ['SAILIR_STRIP_RAWS'] = '1'
os.environ['SAILIR_PACKED_RS'] = '1'
os.environ['SAILIR_SUCCESS_TOTAL'] = '1'
os.environ['SAILIR_ACTION_SELECT'] = 'maxweight'
# SAILIR_BEAM_TOTAL intentionally absent -> (r,s) beam + (r,s) maxweight.

# ── argv injection so beam_search_v7's import-time _cap_incidental_threads()
#    sees the production 8/8 config and applies thread-caps + 8-core pin. It
#    peeks sys.argv for --n-threads/--n-workers; our own parser uses
#    parse_known_args() below so these extra flags are ignored. ──
# CPU count from --v7-cpus (default 8). Peeked from sys.argv BEFORE importing
# beam_search_v7, because its import-time _cap_incidental_threads() reads the
# injected --n-threads/--n-workers to size the affinity pin + thread caps.
def _peek_v7_cpus(default=8):
    for i, a in enumerate(sys.argv):
        if a == '--v7-cpus' and i + 1 < len(sys.argv):
            try:
                return max(1, int(sys.argv[i + 1]))
            except ValueError:
                return default
        if a.startswith('--v7-cpus='):
            try:
                return max(1, int(a.split('=', 1)[1]))
            except ValueError:
                return default
    return default


N_THREADS = N_WORKERS = _peek_v7_cpus()
if '--n-threads' not in sys.argv:
    sys.argv += ['--n-threads', str(N_THREADS)]
if '--n-workers' not in sys.argv:
    sys.argv += ['--n-workers', str(N_WORKERS)]

import argparse
import pickle
import resource
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))   # .../SAILIR_phase2 (repo root) for `sailir`
sys.path.insert(0, str(_HERE))          # .../reduction for sibling modules

# Import beam_search_v7 FIRST. Its module top runs _cap_incidental_threads()
# (thread caps + 8-core affinity pin) BEFORE it (internally) imports numpy/torch,
# so torch's threadpool is created already-pinned to 8 cores.
#
# CRITICAL ORDER: os.sched_setaffinity(0, mask) pins only the CALLING thread and
# any threads created AFTERWARD. If torch (or sailir.classifier, which imports
# torch) is imported BEFORE this, torch spins up its threadpool across ALL cores,
# and the later main-thread-only pin canNOT reclaim those threads -> total CPU
# exceeds RequestCpus -> Condor holds the job. So beam_search_v7 (hence its
# torch) must load before any other torch-importing module. This mirrors the
# standalone beam_search_v7.py order (_cap_incidental_threads() then import torch).
import beam_search_v7 as bs7
from beam_search_v7 import (beam_search_v5, replay_full_expr, _is_success,
                            _target_key, max_w12)
import torch
from sailir import ibp_env
from sailir.topology import Topology
from sailir.ibp_env import (set_prime, set_paper_masters_only, IBPEnvironment,
                            weight)
from sailir.classifier import IBPActionClassifier
from sailir.classifier_nosubs import IBPActionClassifierNoSubs
from beam_search_utils import get_sector_mask


def main():
    p = argparse.ArgumentParser(
        description='in-process v7 onestep worker for hierarchical_reduction.py')
    p.add_argument('--topology', required=True)
    p.add_argument('--integral', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--model-checkpoint', required=True)
    p.add_argument('--beam_width', type=int, default=40)
    p.add_argument('--max_steps', type=int, default=10**6)
    p.add_argument('--prime', type=int, default=1009)
    p.add_argument('--device', default='cpu')
    p.add_argument('-v', '--verbose', action='store_true')
    p.add_argument('--paper-masters-only',
                   action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--resume-from', default=None)
    # Accepted for CLI parity with onestep_worker_v6 (some ignored — v7 fixes them).
    p.add_argument('--n_workers', type=int, default=N_WORKERS)
    p.add_argument('--random-target', action='store_true')
    p.add_argument('--dedup-beam-by-content',
                   action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--beam-sort', default='weight')
    p.add_argument('--no-tabu', dest='tabu', action='store_false', default=True)
    p.add_argument('--no-lazy-rs', dest='lazy_rs', action='store_false', default=True)
    p.add_argument('--iraws-keep-first', type=int, default=50)
    p.add_argument('--no-exprkeyed', dest='use_exprkeyed',
                   action='store_false', default=False)
    p.add_argument('--checkpoint-path', default=None)
    p.add_argument('--no-checkpoint', action='store_true')
    p.add_argument('--checkpoint-interval', type=int, default=100)
    p.add_argument('--checkpoint-time-seconds', type=int, default=300)
    # parse_known_args: tolerate the injected --n-threads/--n-workers flags.
    args, _unknown = p.parse_known_args()

    # ── replicate beam_search_v7.main()'s setup VERBATIM ──────────────────
    torch.set_num_threads(N_THREADS)
    if (os.environ.get('SAILIR_CAP_INTEROP', '1') != '0'
            and (N_THREADS > 1 or N_WORKERS > 1)):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    t0 = time.time()
    topology = Topology.from_dir(args.topology)
    ibp_env.init_from_topology(topology)
    set_prime(args.prime)
    set_paper_masters_only(args.paper_masters_only)
    env = IBPEnvironment()

    # v7 module globals that beam_search_v7.main() sets (replicate exactly).
    bs7._V7_REGISTRY = bs7.IntegralRegistry()
    bs7._V7_PACKED_RS_CACHE = {}
    bs7._PACKED_RS = (os.environ.get('SAILIR_PACKED_RS', '0') == '1')

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

    integral_str = args.integral.strip("'").strip('"')
    start_int = tuple(int(x) for x in integral_str.split(','))
    start_w = weight(start_int)
    start_w12 = (start_w[0], start_w[1])
    bs7._START_TOTAL_KEY = _target_key(start_int)         # SAILIR_SUCCESS_TOTAL=1
    if os.environ.get('SAILIR_STRIP_RAWS', '1') != '0':
        ibp_env.set_raw_strip_threshold(start_w12)
    target_sector = tuple(get_sector_mask(start_int))
    start_expr = {start_int: 1}

    ckpt_path = None
    if args.checkpoint_path and not args.no_checkpoint:
        ckpt_path = args.checkpoint_path

    if args.verbose:
        print(f'[v7-worker] integral I{list(start_int)} weight={start_w12} '
              f'target_sector={target_sector}', flush=True)
        print(f'[v7-worker] beam_width={args.beam_width} max_steps={args.max_steps} '
              f'n_threads={N_THREADS} n_workers={N_WORKERS} '
              f'SUCCESS_TOTAL={bs7._SUCCESS_TOTAL} ACTION_SELECT={bs7._ACTION_SELECT}',
              flush=True)

    beam, best_state = beam_search_v5(
        env, model, start_expr, target_sector, start_w12,
        beam_width=args.beam_width,
        max_steps=args.max_steps,
        device=args.device,
        beam_sort='weight',
        max_actions=900,
        ckpt_path=ckpt_path,
        ckpt_every=args.checkpoint_interval,
        verbose=True,
        resume_from=args.resume_from,
        tabu=True,
        use_incremental_aux=True,
        use_exprkeyed=False,
        ckpt_every_step=False,
        iraws_window=None,
        iraws_keep_first=args.iraws_keep_first,
        lazy_rs=True,
        model_batch_chunk=8,
        top_k=None,
        n_workers=N_WORKERS,
    )

    elapsed = time.time() - t0
    success = _is_success(best_state, target_sector)
    path = list(best_state.path)
    n_steps = len(path)

    # Full expression (passengers + masters) via path replay, exactly as
    # beam_search_v7.main() does for its --output.
    full = replay_full_expr(start_expr, best_state.path, env)
    full_expr = full[0] if full else dict(best_state.expr)

    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    result = {
        'success': success,
        'original_integral': start_int,
        'final_expr': full_expr,
        'path': path,
        'restart_offsets': [],
        'steps': n_steps,
        'time': elapsed,
        'prime': args.prime,
        'peak_memory_kb': peak_rss_kb,
        'best_n_non_masters': best_state.n_non_masters,
        'best_max_w12': max_w12(best_state.expr, target_sector),
        'start_w12': start_w12,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(result, f)

    if args.verbose:
        status = 'SUCCESS' if success else 'INCOMPLETE'
        print(f'[v7-worker] {status} in {elapsed:.2f}s path_len={n_steps} '
              f'nm={best_state.n_non_masters} peak_rss={peak_rss_kb/1024:.0f}MB '
              f'-> {args.output}', flush=True)
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
