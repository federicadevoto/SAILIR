#!/usr/bin/env python3
"""
Async hierarchical reduction with memoization.

Uses onestep_worker with running beam buffer
     for reduced peak memory consumption (only beam_width+1 States alive
     at any time instead of ~800).

Strategy:
1. Maintain a global expression and a cache of reduction results
2. Submit jobs for ALL non-master integrals not in cache or pending
3. As results come in, cache them and apply substitutions to the expression
4. Keep workers saturated - no artificial synchronization points
5. Cache hits avoid redundant work when stragglers produce already-reduced integrals

Usage:
    python hierarchical_reduction.py \
        --integral=1,1,1,1,1,1,-3 \
        --output results/reduction_async.pkl \
        --work-dir /scratch/ibp_async
"""

import sys
import argparse
import pickle
import time
import subprocess
import os
import resource
import signal
import atexit
from pathlib import Path
from collections import defaultdict

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))   # repo root (SAILIR_phase2) for `sailir`
sys.path.insert(0, str(_HERE.parent))          # this dir (reduction/) for siblings

from sailir.ibp_env import IBPEnvironment, set_prime, set_paper_masters_only, is_master, weight, PRIME
from beam_search_utils import get_sector_mask

REPO_DIR = Path(__file__).parent.parent.resolve()
PYTHON_PATH = os.environ.get("SAILIR_PYTHON", sys.executable)  # override with SAILIR_PYTHON env var if Condor workers need a different interpreter


def integral_to_str(integral):
    """Convert integral tuple to string for filenames."""
    return '_'.join(str(x) for x in integral)


def apply_substitutions(expr, cache, prime):
    """Apply all cached substitutions to an expression until no more apply.

    This recursively substitutes any integral that's in the cache with its
    reduced form, until only masters and un-cached integrals remain.
    """
    changed = True
    iterations = 0
    while changed:
        changed = False
        iterations += 1
        new_expr = {}
        for integral, coeff in expr.items():
            if coeff == 0:
                continue
            if integral in cache:
                # Substitute: integral -> cache[integral]
                for sub_int, sub_coeff in cache[integral].items():
                    if sub_coeff == 0:
                        continue
                    new_expr[sub_int] = (new_expr.get(sub_int, 0) + coeff * sub_coeff) % prime
                changed = True
            else:
                new_expr[integral] = (new_expr.get(integral, 0) + coeff) % prime
        expr = {k: v for k, v in new_expr.items() if v != 0}

        # Safety check
        if iterations > 10000:
            print(f"WARNING: apply_substitutions exceeded 10000 iterations")
            break

    return expr


def get_non_masters(expr):
    """Get all non-master integrals in an expression."""
    return {i for i, c in expr.items() if c != 0 and not is_master(i)}


def full_weight(integral):
    """Return (level, r, s) for lex-ordering. level (number of denominators in
    the sector mask) dominates because reductions are hierarchical: an L8
    integral with low (r,s) gates an L7 integral with high (r,s)."""
    level = sum(get_sector_mask(integral))
    r, s = weight(integral)[:2]
    return (level, r, s)


def work_units(integrals):
    """Scalar progress metric: large enough field for each component that
    level dominates r, r dominates s — no collisions across realistic
    pentagon-box ranges (level<=8, r<=20, s<=20)."""
    total = 0
    for i in integrals:
        L, r, s = full_weight(i)
        total += L * 1_000_000 + r * 1000 + s
    return total


def create_condor_submit(work_dir, integral, job_name, output_file,
                         model_checkpoint, beam_width, max_steps, prime,
                         topology_dir,
                         paper_masters_only=True, cpus=1, beam_sort='mixed',
                         checkpoint_path=None, checkpoint_interval=50,
                         checkpoint_time_seconds=300, resume_from=None,
                         dedup_beam_by_content=False,
                         use_delta_worker=False, memory_gb=None,
                         use_v6_worker=False, use_v7_worker=False, v7_cpus=1):
    """Create a Condor submit file for a single-integral one-step reduction.

    PAPER DEFAULTS (matches trianglebox-paper recipe):
      paper_masters_only=True, beam_sort='mixed', dedup_beam_by_content=False.

    If checkpoint_path is set, the worker saves its beam state there every
    checkpoint_interval steps. If resume_from is set, the worker loads that
    checkpoint and continues — used by straggler resubmits to skip work the
    killed 1-CPU worker already did.

    use_delta_worker=True: dispatch to delta_onestep_worker.py instead of
    onestep_worker.py. The delta worker only accepts a smaller set of flags:
    paper_masters_only, beam_width, max_steps, prime, device. n_workers,
    beam_sort, dedup_beam_by_content, checkpoint_path, resume_from are all
    silently ignored (the delta beam search is serial 1-cpu, no checkpoint
    needed because no straggler workflow when stragglers are disabled).

    memory_gb: explicit memory request. If None, falls back to 4 * cpus.
    """

    integral_str = ','.join(str(x) for x in integral)
    if use_v7_worker:
        # The worker runs beam_search_v7 at --n-threads v7_cpus --n-workers v7_cpus
        # and pins itself to that many cores. For v7_cpus>1 the v7_cpus-way enumerate
        # fork-spawn has a rare (~1%) transient that Condor samples slightly above
        # the request, so add 2 cores of headroom. At v7_cpus==1 there is NO fork
        # pool (serial enumerate), so request exactly 1.
        cpus = v7_cpus + (2 if v7_cpus > 1 else 0)
    # Worker now defaults to --paper-masters-only ON. Only emit a flag if we want to disable.
    paper_masters_flag = '' if paper_masters_only else ' --no-paper-masters-only'
    n_workers_flag = f' --n_workers {cpus}' if cpus > 1 else ''
    # Worker now defaults to --beam-sort mixed. Only emit a flag if non-default.
    beam_sort_flag = f' --beam-sort {beam_sort}' if beam_sort != 'mixed' else ''
    cp_flag = (f' --checkpoint-path {checkpoint_path}'
               f' --checkpoint-interval {checkpoint_interval}'
               f' --checkpoint-time-seconds {checkpoint_time_seconds}') if checkpoint_path else ''
    resume_flag = f' --resume-from {resume_from}' if resume_from else ''
    # Worker now defaults to dedup OFF. Only emit a flag if we want to ENABLE dedup.
    dedup_flag = ' --dedup-beam-by-content' if dedup_beam_by_content else ''
    # Schedule by hierarchical weight (level, r, s) — level dominates because
    # reductions are sector-by-sector, so an L8 integral with low (r,s) must
    # clear before its L7 descendants can be finalized. Stragglers (cpus > 1)
    # get a small bonus so a promoted hard target stays ahead of equal-weight
    # 1-CPU jobs.
    level = sum(get_sector_mask(integral))
    r, s = weight(integral)[:2]
    job_priority = level * 1_000_000 + r * 1000 + s + (50 if cpus > 1 else 0)

    # Memory: use explicit --worker-memory-gb as the L=8 (heaviest) request,
    # and scale DOWN for lower levels. Lower-level integrals have far smaller
    # incremental aux (fewer denominators -> fewer cu entries), so giving
    # them 16GB each wastes cluster capacity. Empirically, (8,4) peaks at
    # ~10.6 GB, but (7,*) and (6,*) workers see far less.
    if memory_gb is not None:
        if level >= 8:
            memory = memory_gb
        elif level == 7:
            memory = max(8, memory_gb // 2)
        elif level == 6:
            memory = max(4, memory_gb // 4)
        else:
            memory = 4
        # Straggler resubmits at cpus>1 should be heavy regardless of level
        if cpus > 1:
            memory = max(memory, memory_gb)
    else:
        memory = 4 * cpus
    if use_v7_worker:
        if v7_cpus > 1:
            memory = max(memory, 12)   # floor for the multi-cpu fork-pool worker
        else:
            memory = memory_gb if memory_gb else 4  # flat for 1-cpu serial v7

    if use_v7_worker:
        # onestep_worker_v7.py: runs beam_search_v7 IN-PROCESS at
        # --n-threads v7_cpus --n-workers v7_cpus with ALL v7 settings
        # (SUCCESS_TOTAL=1 total-weight single-step success, (r,s) maxweight
        # action-cap, GNU MKL layer + affinity pin so the worker stays <= request
        # on Condor). v7_cpus==1 => serial, no fork pool. Same CLI as v6.
        worker_script = 'onestep_worker_v7.py'
        worker_args = (f' --topology {topology_dir} --integral=\'{integral_str}\''
                       f' --output {output_file}'
                       f' --model-checkpoint {model_checkpoint}'
                       f' --beam_width {beam_width} --max_steps {max_steps}'
                       f' --prime {prime} --device cpu -v --v7-cpus {v7_cpus}'
                       f'{paper_masters_flag}{resume_flag}')
    elif use_v6_worker:
        # onestep_worker_v6.py: serial 1-cpu, beam_search_v6 underneath
        # (strip-passenger + LAZY_RS + iraws-keep-first=50 + tabu +
        # any()-termination + macro-dedup). Accepts the same CLI as
        # onestep_worker.py and writes orchestrator-compatible result.pkl.
        worker_script = 'onestep_worker_v6.py'
        worker_args = (f' --topology {topology_dir} --integral=\'{integral_str}\''
                       f' --output {output_file}'
                       f' --model-checkpoint {model_checkpoint}'
                       f' --beam_width {beam_width} --max_steps {max_steps}'
                       f' --prime {prime} --device cpu -v'
                       f'{paper_masters_flag}{cp_flag}{resume_flag}')
    elif use_delta_worker:
        # delta_onestep_worker.py: serial 1-cpu, delta-tracking beam search
        # with Cython phase1b + Phase A. CLI surface is smaller — drops
        # checkpoint, resume, n_workers, beam_sort, dedup flags.
        worker_script = 'delta_onestep_worker.py'
        worker_args = (f' --topology {topology_dir} --integral=\'{integral_str}\''
                       f' --output {output_file}'
                       f' --model-checkpoint {model_checkpoint}'
                       f' --beam_width {beam_width} --max_steps {max_steps}'
                       f' --prime {prime} --device cpu -v'
                       f'{paper_masters_flag}')
    else:
        worker_script = 'onestep_worker.py'
        worker_args = (f' --topology {topology_dir} --integral=\'{integral_str}\''
                       f' --output {output_file}'
                       f' --model-checkpoint {model_checkpoint}'
                       f' --beam_width {beam_width} --max_steps {max_steps}'
                       f' --prime {prime} --device cpu -v'
                       f'{paper_masters_flag}{n_workers_flag}{beam_sort_flag}'
                       f'{cp_flag}{resume_flag}{dedup_flag}')

    submit_content = f"""universe = vanilla
executable = {PYTHON_PATH}
arguments = -u {REPO_DIR}/reduction/{worker_script}{worker_args}
output = {work_dir}/logs/{job_name}.out
error = {work_dir}/logs/{job_name}.err
log = {work_dir}/logs/{job_name}.log
request_cpus = {cpus}
request_memory = {memory}GB
request_disk = 1GB
Requirements = (TARGET.KFlops > 3000000)
priority = {job_priority}
+JobFlavour = "workday"
queue
"""

    submit_file = work_dir / f'{job_name}.sub'
    with open(submit_file, 'w') as f:
        f.write(submit_content)

    return submit_file


# NOTE: _compute_job_fields mirrors the per-job computation in
# create_condor_submit above. If you change the worker CLI / memory / priority
# logic there, mirror it here. (Kept separate so create_condor_submit -- the
# straggler-resubmit path -- stays byte-identical and untouched.)
def _compute_job_fields(integral, output_file, model_checkpoint, beam_width,
                        max_steps, prime, topology_dir, paper_masters_only, cpus,
                        beam_sort, checkpoint_path, checkpoint_interval,
                        checkpoint_time_seconds, resume_from,
                        dedup_beam_by_content, use_delta_worker, memory_gb,
                        use_v6_worker, use_v7_worker, v7_cpus):
    """Returns (worker_script, worker_args, cpus, memory, job_priority)."""
    integral_str = ','.join(str(x) for x in integral)
    if use_v7_worker:
        cpus = v7_cpus + (2 if v7_cpus > 1 else 0)
    paper_masters_flag = '' if paper_masters_only else ' --no-paper-masters-only'
    n_workers_flag = f' --n_workers {cpus}' if cpus > 1 else ''
    beam_sort_flag = f' --beam-sort {beam_sort}' if beam_sort != 'mixed' else ''
    cp_flag = (f' --checkpoint-path {checkpoint_path}'
               f' --checkpoint-interval {checkpoint_interval}'
               f' --checkpoint-time-seconds {checkpoint_time_seconds}') if checkpoint_path else ''
    resume_flag = f' --resume-from {resume_from}' if resume_from else ''
    dedup_flag = ' --dedup-beam-by-content' if dedup_beam_by_content else ''
    level = sum(get_sector_mask(integral))
    r, s = weight(integral)[:2]
    job_priority = level * 1_000_000 + r * 1000 + s + (50 if cpus > 1 else 0)
    if memory_gb is not None:
        if level >= 8:
            memory = memory_gb
        elif level == 7:
            memory = max(8, memory_gb // 2)
        elif level == 6:
            memory = max(4, memory_gb // 4)
        else:
            memory = 4
        if cpus > 1:
            memory = max(memory, memory_gb)
    else:
        memory = 4 * cpus
    if use_v7_worker:
        if v7_cpus > 1:
            memory = max(memory, 12)
        else:
            memory = memory_gb if memory_gb else 4
    if use_v7_worker:
        worker_script = 'onestep_worker_v7.py'
        worker_args = (f' --topology {topology_dir} --integral=\'{integral_str}\''
                       f' --output {output_file}'
                       f' --model-checkpoint {model_checkpoint}'
                       f' --beam_width {beam_width} --max_steps {max_steps}'
                       f' --prime {prime} --device cpu -v --v7-cpus {v7_cpus}'
                       f'{paper_masters_flag}{resume_flag}')
    elif use_v6_worker:
        worker_script = 'onestep_worker_v6.py'
        worker_args = (f' --topology {topology_dir} --integral=\'{integral_str}\''
                       f' --output {output_file}'
                       f' --model-checkpoint {model_checkpoint}'
                       f' --beam_width {beam_width} --max_steps {max_steps}'
                       f' --prime {prime} --device cpu -v'
                       f'{paper_masters_flag}{cp_flag}{resume_flag}')
    elif use_delta_worker:
        worker_script = 'delta_onestep_worker.py'
        worker_args = (f' --topology {topology_dir} --integral=\'{integral_str}\''
                       f' --output {output_file}'
                       f' --model-checkpoint {model_checkpoint}'
                       f' --beam_width {beam_width} --max_steps {max_steps}'
                       f' --prime {prime} --device cpu -v'
                       f'{paper_masters_flag}')
    else:
        worker_script = 'onestep_worker.py'
        worker_args = (f' --topology {topology_dir} --integral=\'{integral_str}\''
                       f' --output {output_file}'
                       f' --model-checkpoint {model_checkpoint}'
                       f' --beam_width {beam_width} --max_steps {max_steps}'
                       f' --prime {prime} --device cpu -v'
                       f'{paper_masters_flag}{n_workers_flag}{beam_sort_flag}'
                       f'{cp_flag}{resume_flag}{dedup_flag}')
    return worker_script, worker_args, cpus, memory, job_priority


def create_batch_submit(work_dir, batch, model_checkpoint, beam_width, max_steps,
                        prime, topology_dir, paper_masters_only, beam_sort,
                        checkpoint_interval, checkpoint_time_seconds,
                        dedup_beam_by_content, use_delta_worker, use_v6_worker,
                        use_v7_worker, v7_cpus, memory_gb):
    """ONE submit file for a whole batch -> ONE condor_submit -> one cluster with
    procs 0..N-1 in batch order. This collapses N submit RPCs into 1 (the schedd
    load fix: a 75-job iteration becomes a single round-trip instead of 75).

    batch: list of (integral, job_name, output_file, cpus, resume_from, is_re_entry).
    Common attrs (universe/executable/request_disk/Requirements/+JobFlavour) are
    set once and persist across queue statements; per-job attrs are re-set per
    block. Returns the .sub path.
    """
    parts = [f"""universe = vanilla
executable = {PYTHON_PATH}
request_disk = 1GB
Requirements = (TARGET.KFlops > 3000000)
+JobFlavour = "workday"
"""]
    for (integral, job_name, output_file, cpus, resume_from, _ire) in batch:
        checkpoint_path = str(output_file) + '.checkpoint'
        worker_script, worker_args, jcpus, memory, job_priority = _compute_job_fields(
            integral, output_file, model_checkpoint, beam_width, max_steps, prime,
            topology_dir, paper_masters_only, cpus, beam_sort, checkpoint_path,
            checkpoint_interval, checkpoint_time_seconds, resume_from,
            dedup_beam_by_content, use_delta_worker, memory_gb,
            use_v6_worker, use_v7_worker, v7_cpus)
        parts.append(f"""arguments = -u {REPO_DIR}/reduction/{worker_script}{worker_args}
output = {work_dir}/logs/{job_name}.out
error = {work_dir}/logs/{job_name}.err
log = {work_dir}/logs/{job_name}.log
request_cpus = {jcpus}
request_memory = {memory}GB
priority = {job_priority}
queue
""")
    submit_file = work_dir / f'batch_{batch[0][1]}.sub'
    with open(submit_file, 'w') as f:
        f.write(''.join(parts))
    return submit_file


def build_worker_argv(integral, output_file,
                      model_checkpoint, beam_width, max_steps, prime,
                      topology_dir,
                      paper_masters_only=True, cpus=1, beam_sort='mixed',
                      checkpoint_path=None, checkpoint_interval=50,
                      checkpoint_time_seconds=300, resume_from=None,
                      dedup_beam_by_content=False,
                      use_delta_worker=False,
                      use_v6_worker=False, use_v7_worker=False, v7_cpus=1):
    """Build (worker_script, argv_list, effective_cpus) for a direct (non-Condor)
    subprocess launch of a single-integral one-step reduction.

    This is the --backend local counterpart of create_condor_submit(): same
    flag logic and branching (use_v7_worker / use_v6_worker / use_delta_worker /
    else), but argv_list is a real list of strings suitable for subprocess.Popen
    (no shell quoting), and effective_cpus mirrors create_condor_submit()'s
    v7_cpus-derived cpu count (== request_cpus there) so local CPU-budget
    accounting matches what Condor would have reserved for the same job.

    Deliberately a separate implementation from create_condor_submit() (not a
    shared refactor) so the existing Condor .sub-file generation is provably
    unaffected by this addition.
    """
    integral_str = ','.join(str(x) for x in integral)
    effective_cpus = cpus
    if use_v7_worker:
        effective_cpus = v7_cpus + (2 if v7_cpus > 1 else 0)

    paper_masters_flag = [] if paper_masters_only else ['--no-paper-masters-only']
    n_workers_flag = ['--n_workers', str(effective_cpus)] if effective_cpus > 1 else []
    beam_sort_flag = ['--beam-sort', beam_sort] if beam_sort != 'mixed' else []
    cp_flag = ([] if not checkpoint_path else
              ['--checkpoint-path', str(checkpoint_path),
               '--checkpoint-interval', str(checkpoint_interval),
               '--checkpoint-time-seconds', str(checkpoint_time_seconds)])
    resume_flag = ['--resume-from', str(resume_from)] if resume_from else []
    dedup_flag = ['--dedup-beam-by-content'] if dedup_beam_by_content else []

    # NOTE: --integral is passed as a single '--integral=value' token, not two
    # separate argv elements. integral_str can start with '-' (a negative
    # leading index, e.g. '-1,-1,1,...'); argparse's negative-number heuristic
    # only recognizes plain integers/floats, so a bare '-1,...' token after a
    # separate '--integral' token is misread as an unrecognized option and
    # argparse aborts with "expected one argument". The Condor path sidesteps
    # this the same way via --integral='...' in its shell-quoted arguments=
    # string; here we do it directly since this is a real argv list.
    common_head = ['--topology', str(topology_dir), f'--integral={integral_str}',
                   '--output', str(output_file),
                   '--model-checkpoint', str(model_checkpoint),
                   '--beam_width', str(beam_width), '--max_steps', str(max_steps),
                   '--prime', str(prime), '--device', 'cpu', '-v']

    if use_v7_worker:
        worker_script = 'onestep_worker_v7.py'
        argv = (common_head + ['--v7-cpus', str(v7_cpus)]
                + paper_masters_flag + resume_flag)
    elif use_v6_worker:
        worker_script = 'onestep_worker_v6.py'
        argv = common_head + paper_masters_flag + cp_flag + resume_flag
    elif use_delta_worker:
        worker_script = 'delta_onestep_worker.py'
        argv = common_head + paper_masters_flag
    else:
        worker_script = 'onestep_worker.py'
        argv = (common_head + paper_masters_flag + n_workers_flag + beam_sort_flag
                + cp_flag + resume_flag + dedup_flag)

    return worker_script, argv, effective_cpus


def launch_local_job(work_dir, job_name, worker_script, argv):
    """Launch a worker as a direct subprocess (--backend local). No shell, no
    Condor — stdout/stderr go to the same work_dir/logs/*.out/.err files the
    Condor path would have used. Returns the Popen handle (kept as the opaque
    job handle in `pending`, in place of a Condor cluster id).
    """
    cmd = [PYTHON_PATH, '-u', str(REPO_DIR / 'reduction' / worker_script)] + argv
    out_path = work_dir / 'logs' / f'{job_name}.out'
    err_path = work_dir / 'logs' / f'{job_name}.err'
    out_fh = open(out_path, 'w')
    err_fh = open(err_path, 'w')
    try:
        proc = subprocess.Popen(cmd, stdout=out_fh, stderr=err_fh, cwd=str(REPO_DIR))
    finally:
        # The child has its own dup'd fds; close our copies immediately so we
        # don't leak file descriptors across thousands of jobs in a long run.
        out_fh.close()
        err_fh.close()
    return proc


def kill_local_proc(proc, term_timeout=5.0):
    """Terminate a local worker subprocess, escalating to SIGKILL if it doesn't
    exit promptly. Always reaps (no zombies)."""
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=term_timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    except Exception:
        pass


def submit_condor_job(submit_file):
    """Submit ONE Condor job; return its 'cluster.proc' id ('C.0'). Used by the
    straggler-resubmit path (disabled in production)."""
    try:
        result = subprocess.run(
            ['condor_submit', str(submit_file)],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            # Parse output like: "1 job(s) submitted to cluster 91346."
            import re
            match = re.search(r'cluster\s+(\d+)', result.stdout, re.IGNORECASE)
            if match:
                return f"{match.group(1)}.0"
        return None
    except Exception as e:
        print(f"Error submitting job: {e}")
        return None


def submit_condor_batch(submit_file, expected_n):
    """Submit a multi-job batch .sub. Returns (cluster_id_str, n_submitted).
    condor_submit prints 'N job(s) submitted to cluster C'; the N jobs are procs
    0..N-1 in file order."""
    try:
        result = subprocess.run(
            ['condor_submit', str(submit_file)],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            import re
            m = re.search(r'(\d+)\s+job\(s\)\s+submitted to cluster\s+(\d+)',
                          result.stdout, re.IGNORECASE)
            if m:
                return m.group(2), int(m.group(1))
        return None, 0
    except Exception as e:
        print(f"Error submitting batch: {e}")
        return None, 0


def query_job_start_times(cluster_ids):
    """Query Condor for JobStartDate (unix time) of each cluster_id.

    Returns dict {cluster_id_str: unix_start_time}. Jobs that are still queued
    (have JobStartDate of 0 or undefined) are NOT included in the result, so
    callers can treat them as having zero running time.
    """
    if not cluster_ids:
        return {}
    # Keyed by 'cluster.proc' since batch submits put many jobs in one cluster.
    # Query unique clusters (fewer args) and reconstruct 'C.P' from ProcId.
    clusters = sorted({str(c).split('.')[0] for c in cluster_ids})
    try:
        result = subprocess.run(
            ['condor_q'] + clusters + ['-af', 'ClusterId', 'ProcId', 'JobStartDate'],
            capture_output=True, text=True, timeout=30,
        )
        out = {}
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) != 3:
                continue
            cid, pid, start = parts
            if start in ('undefined', '0'):
                continue  # not yet started
            try:
                out[f"{cid}.{pid}"] = int(start)
            except ValueError:
                continue
        return out
    except Exception as e:
        print(f"Error querying JobStartDate: {e}")
        return {}


def check_job_status(cluster_ids):
    """Check status of Condor jobs. Returns set of completed cluster IDs."""
    if not cluster_ids:
        return set()

    try:
        result = subprocess.run(
            ['condor_q'] + list(cluster_ids) + ['-format', '%d\n', 'ClusterId'],
            capture_output=True, text=True, timeout=30
        )
        running = set(result.stdout.strip().split('\n')) if result.stdout.strip() else set()
        return set(cluster_ids) - running
    except Exception as e:
        print(f"Error checking job status: {e}")
        return set()


def main():
    parser = argparse.ArgumentParser(description='Async hierarchical reduction with memoization (v13)')
    parser.add_argument('--topology', type=str, required=True,
                        help='Path to topology_input/<family>/ directory; '
                             'passed through to each onestep_worker submission.')
    parser.add_argument('--integral', type=str, required=True,
                        help='Starting integral indices (comma-separated)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output pickle file for reduction result')
    parser.add_argument('--work-dir', type=str, required=True,
                        help='Working directory for intermediate files')
    parser.add_argument('--model-checkpoint', type=str,
                        default=str(REPO_DIR / 'checkpoints/best_model.pt'),
                        help='Path to model checkpoint')
    parser.add_argument('--beam_width', type=int, default=20,
                        help='Beam width for action search')
    parser.add_argument('--max_steps', type=int, default=10**15,
                        help='Max steps per integral (effectively unlimited)')
    parser.add_argument('--prime', type=int, default=1009,
                        help='Prime for modular arithmetic')
    parser.add_argument('--check-interval', type=int, default=5,
                        help='Seconds between job status checks')
    parser.add_argument('--max-concurrent', type=int, default=10000,
                        help='Maximum concurrent Condor jobs')
    parser.add_argument('--backend', type=str, default='condor',
                        choices=['condor', 'local'],
                        help="Execution backend for worker jobs. 'condor' "
                             '(default): submit via condor_submit/condor_q/'
                             "condor_rm, unchanged. 'local': launch workers "
                             'directly via subprocess.Popen on this machine, '
                             'with admission gated by --max-cpus instead of a '
                             'Condor scheduler (for boxes with no scheduler).')
    parser.add_argument('--max-cpus', type=int, default=120,
                        help='Maximum total CPUs to use concurrently for '
                             '--backend local worker jobs (sum of each '
                             "pending job's cpu count). Ignored for "
                             '--backend condor (Condor handles admission '
                             'there). Set below the machine total core count '
                             'on shared boxes.')
    # PAPER DEFAULT: ON (trianglebox-paper recipe).
    parser.add_argument('--paper-masters-only', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Reduce to paper masters only (no corner integrals). '
                             'Default: ON (trianglebox paper recipe).')
    parser.add_argument('--straggler-timeout', type=int, default=3600,
                        help='Seconds of actual Condor RUN time (excluding queue wait) before a '
                             'job is considered a straggler (default 60 min)')
    parser.add_argument('--straggler-cpus', type=int, default=8,
                        help='CPUs to allocate when resubmitting stragglers')
    parser.add_argument('--straggler2-timeout', type=int, default=10800,
                        help='Seconds of actual Condor RUN time on an 8-CPU straggler '
                             'before second escalation kicks in (default 3h)')
    parser.add_argument('--straggler2-cpus', type=int, default=16,
                        help='CPUs to allocate when resubmitting straggler2 (wider '
                             'parallelism for genuinely stuck integrals)')
    parser.add_argument('--straggler2-beam-width', type=int, default=40,
                        help='Beam width for straggler2 worker (wider beam to escape '
                             'model-stuck states; combined with --worker-dedup-beam-'
                             'by-content this gives real diversity)')
    parser.add_argument('--checkpoint-interval', type=int, default=50,
                        help='Steps between beam-search checkpoints (saved next to '
                             'each worker output). Straggler resubmits use the '
                             'previous checkpoint to skip work already done.')
    parser.add_argument('--checkpoint-time-seconds', type=int, default=300,
                        help='Also checkpoint if this many seconds have elapsed; '
                             'protects against very large per-step times.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Create submit files but do not submit')
    # PAPER DEFAULT: mixed (trianglebox-paper recipe — two parallel sub-beams,
    # one sorted by max-weight, one by total sum of weights).
    parser.add_argument('--beam-sort', type=str, default='mixed',
                        choices=['weight', 'nterms', 'score', 'totalweight', 'mixed'],
                        help='Beam sort key for worker jobs. Default: mixed '
                             '(trianglebox paper recipe).')
    # PAPER DEFAULT: dedup OFF (trianglebox-paper recipe).
    parser.add_argument('--worker-dedup-beam-by-content',
                        action=argparse.BooleanOptionalAction, default=False,
                        help='Beam dedup in worker jobs. Default: OFF '
                             '(trianglebox paper recipe). When ON, workers receive '
                             '--dedup-beam-by-content (resolved_subs-fingerprint key) '
                             'and the search keeps only one beam slot per distinct '
                             'fingerprint. Turn ON for hard integrals like (8,4) '
                             'pentagon-box; see memory/sailir_84_full_resolved_subs_dedup.md.')
    parser.add_argument('--use-delta-worker', action='store_true',
                        help='Dispatch workers to delta_onestep_worker.py '
                             '(serial 1-cpu delta-tracking + Cython). Drops '
                             'flags the delta worker does not accept '
                             '(n_workers, beam_sort, dedup, checkpoint, '
                             'resume). Use with large --straggler-timeout '
                             'and --straggler2-timeout to disable the '
                             'straggler escalation entirely.')
    parser.add_argument('--use-v6-worker', action='store_true',
                        help='Dispatch workers to onestep_worker_v6.py '
                             '(serial 1-cpu beam_search_v6 — strip-passenger '
                             '+ LAZY_RS + iraws-keep-first=50 + tabu + '
                             'any() termination + macro-dedup). '
                             'Drops the n_workers/dedup/beam_sort flags. '
                             'Empirically 9.7\u00d7 faster + 10.9\u00d7 lower memory '
                             'than the delta worker on the canonical '
                             'pentagonbox long-runner integral.')
    parser.add_argument('--use-v7-worker', action='store_true',
                        help='Dispatch workers to onestep_worker_v7.py, which runs '
                             'the CURRENT beam_search_v7.py IN-PROCESS at '
                             '--n-threads/--n-workers = --v7-cpus with ALL v7 '
                             'settings (SUCCESS_TOTAL total-weight single-step '
                             'success, (r,s) maxweight action-cap, GNU MKL + '
                             'affinity pin).')
    parser.add_argument('--v7-cpus', type=int, default=1,
                        help='CPUs per v7 worker (= n-threads = n-workers). '
                             'Default 1 (serial, NO fork pool, single torch '
                             'thread, request_cpus=1, flat 4GB) -- the production '
                             'setting: many cheap workers fanned out wide. Set 8 '
                             'only for an 8/8 fork pool (request_cpus=10, ~22GB '
                             'peak) on a few genuinely heavy long-runners.')
    parser.add_argument('--worker-memory-gb', type=int, default=None,
                        help='Override per-worker memory request (GB). '
                             'Default scales with cpus (4 * cpus). For the '
                             'delta worker on heavy targets like (8,5), set '
                             'to 16 or 32 to leave room for the aux growth.')
    parser.add_argument('--resume', action='store_true',
                        help='Resume: on startup, scan work_dir/results/*.pkl and '
                             'load each completed worker as a cache entry. Then '
                             'apply substitutions to start_expr and continue the '
                             'orchestrator loop. Drops any pending jobs (will be '
                             'resubmitted). Use after stopping the orchestrator + '
                             'killing all its workers (condor_rm).')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Round-2 resume: load a PRIOR round\'s cache '
                             'READ-ONLY from <dir> (uses <dir>/replay_state.pkl '
                             'if present, else scans <dir>/results/*.pkl), while '
                             'writing all NEW worker results to --work-dir. The '
                             'prior dir is never modified. Combine with '
                             '--reduce-only to re-reduce a specific subset.')
    parser.add_argument('--reduce-only', type=str, default=None,
                        help='Path to a text file of integrals (one comma-'
                             'separated tuple per line). Restrict THIS round to '
                             'reducing exactly these (plus any genuinely-new '
                             'descendants): the initial expr is seeded to their '
                             'sum and they are dropped from the resumed cache so '
                             'they re-submit. Everything not reachable from this '
                             'list is never entered into expr (set aside).')
    args = parser.parse_args()

    print('='*70)
    print('Async Hierarchical Reduction with Memoization (v13)')
    print('='*70)
    print(f'Config:')
    # Args that the v7 in-process worker IGNORES (it hardcodes its own probe
    # recipe). Flag them so the dump is not misleading (e.g. beam_sort says
    # "mixed" here but the worker always runs beam_sort='weight').
    _v7_overridden = {'beam_sort', 'worker_dedup_beam_by_content', 'beam_width',
                      'straggler_cpus', 'straggler2_cpus', 'straggler2_beam_width'}
    for k, v in vars(args).items():
        note = '   [IGNORED by v7 worker — see note below]' \
            if (getattr(args, 'use_v7_worker', False) and k in _v7_overridden) else ''
        print(f'  {k}: {v}{note}')
    if getattr(args, 'use_v7_worker', False):
        print()
        print('  --- v7 worker FIXED settings (what the search ACTUALLY uses) ---')
        print('  worker            : onestep_worker_v7.py (in-process beam_search_v7)')
        print('  beam_sort         : weight')
        print('  n_threads         : 8        n_workers: 8   (8-core affinity pin)')
        print('  max_actions       : 900      model_batch_chunk: 8')
        print('  tabu              : True     use_exprkeyed: False')
        print('  iraws_keep_first  : 50       lazy_rs: True   use_incremental_aux: True')
        print('  SAILIR_SUCCESS_TOTAL=1  SAILIR_ACTION_SELECT=maxweight  '
              'SAILIR_BEAM_TOTAL=unset')
        print('  SAILIR_PACKED_RS=1  SAILIR_STRIP_RAWS=1  SAILIR_END_OF_STEP_TRIM=1  '
              'SAILIR_TABU_CAP=0')
        print('  (matches submit_v7_successonly_probes.sh exactly)')
    print()

    # Setup
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / 'logs').mkdir(exist_ok=True)
    (work_dir / 'results').mkdir(exist_ok=True)

    # Configure topology first so ibp_env globals are populated.
    from sailir.topology import Topology
    from sailir import ibp_env
    topology = Topology.from_dir(args.topology)
    ibp_env.init_from_topology(topology)
    set_prime(args.prime)
    if args.paper_masters_only:
        set_paper_masters_only(True)

    env = IBPEnvironment()

    # Parse starting integral
    starting_integral = tuple(int(x) for x in args.integral.split(','))
    print(f'Starting integral: I{list(starting_integral)}')
    print(f'Starting weight: {weight(starting_integral)}')
    print()

    # State
    expr = {starting_integral: 1}  # Current expression (linear combo of integrals)
    cache = {}  # integral -> reduced expression (memoization)
    # integral -> (cluster_id, output_file, submit_time, cpus). For
    # --backend local, `cluster_id` is an OPAQUE job handle: a Condor cluster-id
    # string for --backend condor, a live subprocess.Popen object for local.
    pending = {}
    straggler_integrals = set()  # integrals that have been resubmitted as stragglers
    straggler2_integrals = set()  # integrals that have hit the second-level escalation

    # --backend local: clean up any still-running worker subprocesses if the
    # orchestrator exits for ANY reason (normal return, exception, Ctrl-C,
    # SIGTERM) — critical on a shared machine so a stopped orchestrator never
    # leaves orphaned workers eating cores. Every live local Popen is already
    # sitting in `pending`'s first slot, so no separate registry is needed.
    # Known gap: this cannot catch SIGKILL or a segfault of the orchestrator
    # itself (uncatchable/unreachable) — accepted limitation, not fixed here.
    if args.backend == 'local':
        def _cleanup_local_workers():
            for cluster_id, _output_file, _submit_time, _cpus in pending.values():
                kill_local_proc(cluster_id)
        atexit.register(_cleanup_local_workers)

        def _signal_handler(signum, frame):
            _cleanup_local_workers()
            sys.exit(1)
        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

    # --resume: scan work_dir/results/*.pkl, load each successful worker output
    # as a cache entry. Apply substitutions to start_expr to recover current expr.
    # Pending is dropped (any in-flight workers should be condor_rm'd before
    # restarting the orchestrator). This is the orchestrator-level resume; the
    # worker-level resume (per-integral) is independent.
    if args.resume:
        import glob
        t_resume = time.time()
        results_dir = work_dir / 'results'
        pkl_files = sorted(glob.glob(str(results_dir / '*.pkl')))
        # Skip per-worker checkpoint files (they end in .pkl.checkpoint, glob
        # matches both — filter explicitly).
        pkl_files = [p for p in pkl_files if not p.endswith('.checkpoint')]
        print(f'[RESUME] Loading {len(pkl_files)} worker pickles from {results_dir} ...',
              flush=True)
        n_loaded = 0
        n_skipped = 0
        for i, pf in enumerate(pkl_files):
            try:
                with open(pf, 'rb') as f:
                    r = pickle.load(f)
            except Exception as e:
                n_skipped += 1
                continue
            integ = r.get('original_integral')
            if integ is None:
                n_skipped += 1
                continue
            if r.get('success'):
                cache[integ] = r.get('final_expr', r.get('expr', {integ: 1}))
            else:
                # Failed worker: identity cache so we don't re-try and loop.
                cache[integ] = {integ: 1}
            n_loaded += 1
            if (i + 1) % 5000 == 0:
                print(f'  [RESUME]   ... {i+1}/{len(pkl_files)} loaded', flush=True)
        print(f'[RESUME] Loaded {n_loaded} cache entries ({n_skipped} skipped) '
              f'in {time.time()-t_resume:.1f}s', flush=True)
        # Apply all cached substitutions to start_expr to recover current state.
        t_apply = time.time()
        expr = apply_substitutions(expr, cache, args.prime)
        print(f'[RESUME] Applied substitutions to expr in {time.time()-t_apply:.1f}s; '
              f'|expr|={len(expr)}, |cache|={len(cache)}', flush=True)

    # --resume-from: round-2 mode. Load a PRIOR round's cache read-only; new
    # results still write to work_dir. Prefer the prior dir's replay_state.pkl
    # (one fast load) over re-scanning all of results/.
    if args.resume_from:
        import glob
        t_rf = time.time()
        src = Path(args.resume_from)
        replay = src / 'replay_state.pkl'
        if replay.exists():
            with open(replay, 'rb') as f:
                st = pickle.load(f)
            loaded = st['cache'] if (isinstance(st, dict) and 'cache' in st) else st
            cache.update(loaded)
            print(f'[RESUME-FROM] Loaded {len(loaded)} cache entries from '
                  f'{replay} in {time.time()-t_rf:.1f}s', flush=True)
        else:
            results_dir = src / 'results'
            pkl_files = [p for p in sorted(glob.glob(str(results_dir / '*.pkl')))
                         if not p.endswith('.checkpoint')]
            print(f'[RESUME-FROM] Scanning {len(pkl_files)} pickles from '
                  f'{results_dir} ...', flush=True)
            for pf in pkl_files:
                try:
                    with open(pf, 'rb') as f:
                        r = pickle.load(f)
                except Exception:
                    continue
                integ = r.get('original_integral')
                if integ is None:
                    continue
                if r.get('success'):
                    cache[integ] = r.get('final_expr', r.get('expr', {integ: 1}))
                else:
                    cache[integ] = {integ: 1}
            print(f'[RESUME-FROM] Loaded {len(cache)} cache entries in '
                  f'{time.time()-t_rf:.1f}s', flush=True)
        expr = apply_substitutions(expr, cache, args.prime)

    # --reduce-only: restrict this round to a target list by DROPPING those
    # integrals from the (resumed) cache so they re-submit. Seed expr from the
    # REAL start integral replayed through the (target-dropped) cache — NOT a
    # unit-coefficient bag of the targets.
    #
    # BUG FIX: the old seed `{ig: 1 for ig in targets}` gave every target a fake
    # coefficient of 1. With those bogus weights, a target's coefficient could
    # cancel to 0 mod prime via another substitution; the orchestrator's
    # "cancel pending jobs whose integral is no longer in expr" logic then killed
    # it as "no longer needed", it was never reduced, and the run falsely
    # reported "0 non-masters / SUCCESS" — on the unit bag, not the real start.
    # Seeding from the real start gives expr the TRUE coefficients, so a target
    # leaves expr only when it genuinely doesn't contribute, the per-iteration
    # non-master count is an honest live status, and the final 0-non-masters
    # report is a true statement about the start integral.
    if args.reduce_only:
        targets = []
        with open(args.reduce_only) as f:
            for line in f:
                line = line.strip()
                if line:
                    targets.append(tuple(int(x) for x in line.split(',')))
        for ig in targets:
            cache.pop(ig, None)
        expr = apply_substitutions({starting_integral: 1}, cache, args.prime)
        n_nm_seed = len(get_non_masters(expr))
        print(f'[REDUCE-ONLY] {len(targets)} targets dropped from cache; expr '
              f'seeded from REAL start replay (true coefficients); '
              f'|expr|={len(expr)}, non-masters={n_nm_seed}, '
              f'|cache|={len(cache)}', flush=True)

    # integral -> path of its most-recent worker's .checkpoint file. Persists
    # across pending lifecycle (does NOT get cleared by obsolete-cancel) so a
    # re-entering straggler integral can resume from its last saved beam state.
    last_checkpoint = {}

    # Rolling (timestamp, work_units) window for ETA estimation in status print.
    work_history = []
    depth = {}  # integral -> dependency depth (for reporting)
    parent = {}  # integral -> parent integral that discovered it
    ideal_finish = {}  # integral -> earliest possible finish time on critical path

    total_jobs = 0
    total_steps = 0
    total_worker_time = 0.0  # Cumulative worker runtime (excludes queue wait)
    cache_hits = 0
    stragglers_resubmitted = 0
    max_worker_memory_kb = 0  # Peak raw memory across all workers
    max_worker_memory_per_cpu_kb = 0  # Peak per-CPU memory
    start_time = time.time()

    iteration = 0
    while True:
        iteration += 1

        # Apply all cached substitutions to the expression
        old_size = len(expr)
        expr = apply_substitutions(expr, cache, args.prime)

        # Count cache hits (integrals that were substituted)
        # This is approximate - we count how many integrals were removed

        # Cancel pending jobs whose integral is no longer in expr (its
        # coefficient was zeroed mod prime by another substitution). Kill
        # both IDLE and RUNNING workers — a running worker on a substituted-
        # out integral is doing redundant work that won't be inserted back
        # into the live expression. (Earlier policy spared running workers
        # for cache short-circuit benefits, but it was inconsistent: when
        # the JobStartDate query timed out it killed running workers anyway,
        # so different iterations applied different policies.)
        if pending:
            # Collect the obsolete jobs' cluster.proc ids and remove them in ONE
            # condor_rm (a single schedd round-trip instead of one per job).
            obsolete_ids = []
            obsolete_integrals = []
            for integral in list(pending.keys()):
                if integral in expr:
                    continue  # still needed
                cluster_id, output_file, _, _ = pending[integral]
                if output_file.exists():
                    continue  # result already on disk — let normal handling take it
                if not cluster_id:
                    continue
                if args.backend == 'local':
                    kill_local_proc(cluster_id)
                    del pending[integral]
                    obsolete_integrals.append(integral)
                else:
                    obsolete_ids.append(str(cluster_id))
                    obsolete_integrals.append(integral)
            if obsolete_ids:
                try:
                    subprocess.run(['condor_rm'] + obsolete_ids,
                                   capture_output=True, text=True, timeout=120)
                except Exception:
                    pass
                for integral in obsolete_integrals:
                    if integral in pending:
                        del pending[integral]
            if obsolete_integrals:
                print(f"[Iter {iteration}] Cancelled {len(obsolete_integrals)} pending jobs "
                      f"(idle+running) whose integrals are no longer needed", flush=True)

        # Find non-masters that need reduction
        non_masters = get_non_masters(expr)

        # Integrals to submit: non-masters not in cache and not pending
        to_submit = non_masters - set(cache.keys()) - set(pending.keys())

        # Limit concurrent jobs
        available_slots = args.max_concurrent - len(pending)
        if available_slots < len(to_submit):
            # Prioritize by full hierarchical weight (level, r, s); level
            # dominates so upstream sectors clear before downstream ones.
            to_submit = sorted(to_submit, key=lambda i: tuple(-x for x in full_weight(i)))
            to_submit = set(to_submit[:available_slots])

        # Submit new jobs: build the WHOLE iteration's batch, then ONE
        # condor_submit (one cluster, procs 0..N-1). This collapses a 75-job
        # iteration into a single schedd round-trip instead of 75 -- the fix for
        # the schedd saturation that a big cache's spiky submit rate caused.
        newly_submitted = 0
        batch = []  # condor path: accumulated for one batch submit after loop
        # --backend local CPU-budget admission gate: running tally of CPUs
        # already committed to `pending` local jobs, incremented as we admit
        # more this iteration. Composes with (doesn't replace) the job-count
        # cap (--max-concurrent) already applied above via available_slots.
        # Integrals skipped here stay in non_masters/cache-miss next
        # iteration (they're simply not deleted from anything), so they're
        # retried once a slot frees up.
        cpus_in_use_local = (sum(c for (_, _, _, c) in pending.values())
                             if args.backend == 'local' else 0)
        for integral in sorted(to_submit, key=lambda i: tuple(-x for x in full_weight(i))):
            if integral not in depth:
                depth[integral] = 0  # Top-level integral from initial expression

            # If this integral was previously promoted to a straggler but then
            # left `pending` (e.g. obsolete-cancelled while idle in the 8-CPU
            # queue), the original straggler-promotion path at the 60-min
            # timeout will refuse to re-promote it (the `not in
            # straggler_integrals` guard blocks it), so a fresh 1-CPU resubmit
            # would run forever. Promote straight to N CPUs and resume from
            # the last saved checkpoint if one exists.
            is_re_entry = integral in straggler_integrals
            if is_re_entry:
                cpus = args.straggler_cpus
                job_name = f"straggler_{total_jobs}_{integral_to_str(integral)}"
                prev_cp = last_checkpoint.get(integral)
                resume_from = prev_cp if (prev_cp and Path(prev_cp).exists()) else None
            else:
                cpus = 1
                job_name = f"async_{total_jobs}_{integral_to_str(integral)}"
                resume_from = None

            if (args.backend == 'local'
                    and cpus_in_use_local + cpus > args.max_cpus):
                continue  # over budget this iteration -- retry next loop

            output_file = work_dir / 'results' / f'{job_name}.pkl'
            last_checkpoint[integral] = str(output_file) + '.checkpoint'

            if args.backend == 'local':
                worker_script, argv, eff_cpus = build_worker_argv(
                    integral, output_file,
                    args.model_checkpoint, args.beam_width, args.max_steps, args.prime,
                    topology_dir=args.topology,
                    paper_masters_only=args.paper_masters_only, cpus=cpus,
                    beam_sort=args.beam_sort,
                    checkpoint_path=last_checkpoint[integral],
                    checkpoint_interval=args.checkpoint_interval,
                    checkpoint_time_seconds=args.checkpoint_time_seconds,
                    resume_from=resume_from,
                    dedup_beam_by_content=args.worker_dedup_beam_by_content,
                    use_delta_worker=args.use_delta_worker,
                    use_v6_worker=args.use_v6_worker,
                    use_v7_worker=args.use_v7_worker, v7_cpus=args.v7_cpus,
                )
                if args.dry_run:
                    print(f"  [DRY-RUN] would launch: {worker_script} {' '.join(argv)}")
                    pending[integral] = (None, output_file, time.time(), eff_cpus)
                    total_jobs += 1
                    newly_submitted += 1
                else:
                    proc = launch_local_job(work_dir, job_name, worker_script, argv)
                    pending[integral] = (proc, output_file, time.time(), eff_cpus)
                    cpus_in_use_local += eff_cpus
                    total_jobs += 1
                    newly_submitted += 1
                    if is_re_entry:
                        print(f"  RE-ENTRY: Re-launched I{list(integral)} (pid={proc.pid}) "
                              f"with {eff_cpus} CPUs "
                              f"(resume={'yes' if resume_from else 'no'})", flush=True)
                continue  # don't accumulate into condor batch

            # Condor path: accumulate for one batch condor_submit after loop
            batch.append((integral, job_name, output_file, cpus, resume_from, is_re_entry))
            total_jobs += 1   # reserve the async id (job_name embeds it)

        if batch and args.dry_run:
            for (integral, job_name, output_file, cpus, resume_from, is_re_entry) in batch:
                pending[integral] = (None, output_file, time.time(), cpus)
                newly_submitted += 1
        elif batch:
            submit_file = create_batch_submit(
                work_dir, batch,
                args.model_checkpoint, args.beam_width, args.max_steps, args.prime,
                topology_dir=args.topology,
                paper_masters_only=args.paper_masters_only,
                beam_sort=args.beam_sort,
                checkpoint_interval=args.checkpoint_interval,
                checkpoint_time_seconds=args.checkpoint_time_seconds,
                dedup_beam_by_content=args.worker_dedup_beam_by_content,
                use_delta_worker=args.use_delta_worker,
                use_v6_worker=args.use_v6_worker,
                use_v7_worker=args.use_v7_worker, v7_cpus=args.v7_cpus,
                memory_gb=args.worker_memory_gb,
            )
            cluster_id, n_submitted = submit_condor_batch(submit_file, len(batch))
            if cluster_id and n_submitted == len(batch):
                for proc, (integral, job_name, output_file, cpus, resume_from, is_re_entry) in enumerate(batch):
                    pending[integral] = (f"{cluster_id}.{proc}", output_file, time.time(), cpus)
                    newly_submitted += 1
                    if is_re_entry:
                        print(f"  RE-ENTRY: Re-submitted I{list(integral)} with {cpus} CPUs "
                              f"(resume={'yes' if resume_from else 'no'})", flush=True)
            else:
                # Whole batch not accepted -> none added to pending; they stay in
                # to_submit and retry next iteration (correctness preserved).
                print(f"  WARNING: batch submit failed (cluster={cluster_id}, "
                      f"got {n_submitted}/{len(batch)}); will retry next iteration")

        if newly_submitted > 0:
            print(f"[Iter {iteration}] Submitted {newly_submitted} jobs")

        # Check if done
        if not pending and not to_submit:
            print(f"\n[Iter {iteration}] All done!")
            break

        if args.dry_run:
            print(f"[Iter {iteration}] Dry run - stopping after job creation")
            break

        # Wait a bit
        time.sleep(args.check_interval)

        # Check for stragglers (jobs that have been RUNNING too long — queue/
        # idle wait time is excluded). Resubmit them with more CPUs.
        current_time = time.time()
        # Skip the per-iteration condor_q if: local backend (no queue to query)
        # OR stragglers effectively disabled (huge timeouts, the production
        # setting) — in both cases it's a schedd round-trip that buys nothing.
        stragglers_enabled = (args.straggler_timeout < 10**8
                              or args.straggler2_timeout < 10**8)
        if args.backend == 'local' or not stragglers_enabled:
            start_times = {}
        else:
            pending_cluster_ids = [cid for (cid, _, _, _) in pending.values() if cid]
            start_times = query_job_start_times(pending_cluster_ids)
        for integral, (cluster_id, output_file, submit_time, cpus) in list(pending.items()):
            if args.backend == 'local':
                job_runtime = current_time - submit_time
            else:
                # Job runtime is wall time since Condor *started* executing the
                # job. If still queued (no JobStartDate yet) treat runtime as 0
                # so the straggler logic only ever fires on jobs that actually ran.
                start = start_times.get(str(cluster_id)) if cluster_id else None
                job_runtime = (current_time - start) if start else 0
            # Only resubmit as straggler if: running too long, not already a straggler, and using single CPU
            if (job_runtime > args.straggler_timeout and
                integral not in straggler_integrals and
                cpus == 1):
                # Kill the slow job
                if args.backend == 'local':
                    print(f"  Killed straggler job (pid={cluster_id.pid})", flush=True)
                    kill_local_proc(cluster_id)
                elif cluster_id:
                    try:
                        kill_result = subprocess.run(['condor_rm', str(cluster_id)], capture_output=True, text=True, timeout=10)
                        if kill_result.returncode == 0:
                            print(f"  Killed straggler job {cluster_id}", flush=True)
                        else:
                            # Job might have already finished on its own
                            print(f"  Job {cluster_id} already gone (completed or removed)", flush=True)
                    except Exception as e:
                        print(f"  WARNING: condor_rm {cluster_id} exception: {e}", flush=True)
                else:
                    print(f"  WARNING: No cluster_id for straggler I{list(integral)}", flush=True)

                # Remove old output file if it exists (partial)
                if output_file.exists():
                    try:
                        output_file.unlink()
                    except:
                        pass

                # Resubmit with more CPUs, resuming from the killed worker's
                # checkpoint if one exists (saves the ~60 min of beam search
                # the killed 1-CPU job already did).
                job_name = f"straggler_{total_jobs}_{integral_to_str(integral)}"
                new_output_file = work_dir / 'results' / f'{job_name}.pkl'
                prev_cp = last_checkpoint.get(integral) or (str(output_file) + '.checkpoint')
                resume_from = prev_cp if Path(prev_cp).exists() else None
                new_checkpoint = str(new_output_file) + '.checkpoint'
                last_checkpoint[integral] = new_checkpoint

                if args.backend == 'local':
                    worker_script, argv, eff_cpus = build_worker_argv(
                        integral, new_output_file,
                        args.model_checkpoint, args.beam_width, args.max_steps, args.prime,
                        topology_dir=args.topology,
                        paper_masters_only=args.paper_masters_only,
                        cpus=args.straggler_cpus, beam_sort=args.beam_sort,
                        checkpoint_path=new_checkpoint,
                        checkpoint_interval=args.checkpoint_interval,
                        checkpoint_time_seconds=args.checkpoint_time_seconds,
                        resume_from=resume_from,
                        dedup_beam_by_content=args.worker_dedup_beam_by_content,
                        use_delta_worker=args.use_delta_worker,
                        use_v6_worker=args.use_v6_worker,
                        use_v7_worker=args.use_v7_worker, v7_cpus=args.v7_cpus,
                    )
                    new_proc = launch_local_job(work_dir, job_name, worker_script, argv)
                    pending[integral] = (new_proc, new_output_file, time.time(), eff_cpus)
                    straggler_integrals.add(integral)
                    stragglers_resubmitted += 1
                    total_jobs += 1
                    print(f"  STRAGGLER: Re-launched I{list(integral)} (pid={new_proc.pid}) "
                          f"with {eff_cpus} CPUs (was running {job_runtime/60:.1f} min)")
                    continue

                submit_file = create_condor_submit(
                    work_dir, integral, job_name, new_output_file,
                    args.model_checkpoint, args.beam_width, args.max_steps, args.prime,
                    topology_dir=args.topology,
                    paper_masters_only=args.paper_masters_only,
                    cpus=args.straggler_cpus, beam_sort=args.beam_sort,
                    checkpoint_path=new_checkpoint,
                    checkpoint_interval=args.checkpoint_interval,
                    checkpoint_time_seconds=args.checkpoint_time_seconds,
                    resume_from=resume_from,
                    dedup_beam_by_content=args.worker_dedup_beam_by_content,
                    use_delta_worker=args.use_delta_worker,
                    use_v6_worker=args.use_v6_worker,
                use_v7_worker=args.use_v7_worker, v7_cpus=args.v7_cpus,
                    memory_gb=args.worker_memory_gb,
                )

                new_cluster_id = submit_condor_job(submit_file)
                if new_cluster_id:
                    pending[integral] = (new_cluster_id, new_output_file, time.time(), args.straggler_cpus)
                    straggler_integrals.add(integral)
                    stragglers_resubmitted += 1
                    total_jobs += 1
                    print(f"  STRAGGLER: Resubmitted I{list(integral)} with {args.straggler_cpus} CPUs "
                          f"(was running {job_runtime/60:.1f} min)")
                else:
                    print(f"  WARNING: Failed to resubmit straggler I{list(integral)}")

        # Second-level escalation: an integral that has been an 8-CPU straggler
        # for --straggler2-timeout (default 3h) and is STILL running gets
        # promoted to --straggler2-cpus (16) + --straggler2-beam-width (40) +
        # dedup ON. Resumes from the latest checkpoint to preserve progress.
        for integral, (cluster_id, output_file, submit_time, cpus) in list(pending.items()):
            if args.backend == 'local':
                job_runtime = current_time - submit_time
            else:
                start = start_times.get(str(cluster_id)) if cluster_id else None
                job_runtime = (current_time - start) if start else 0
            if (job_runtime > args.straggler2_timeout
                and integral in straggler_integrals
                and integral not in straggler2_integrals
                and cpus == args.straggler_cpus):
                # Kill the 8-CPU job
                if args.backend == 'local':
                    print(f"  Killed straggler2 job (pid={cluster_id.pid})", flush=True)
                    kill_local_proc(cluster_id)
                elif cluster_id:
                    try:
                        kill_result = subprocess.run(['condor_rm', str(cluster_id)],
                                                     capture_output=True, text=True, timeout=10)
                        if kill_result.returncode == 0:
                            print(f"  Killed straggler2 job {cluster_id}", flush=True)
                    except Exception as e:
                        print(f"  WARNING: condor_rm {cluster_id} exception: {e}", flush=True)

                # Remove partial output if any
                if output_file.exists():
                    try: output_file.unlink()
                    except: pass

                # Resubmit at higher level: wider beam + more CPUs + dedup on.
                # Resume from last_checkpoint so we don't lose the 3h of work.
                job_name = f"straggler2_{total_jobs}_{integral_to_str(integral)}"
                new_output_file = work_dir / 'results' / f'{job_name}.pkl'
                prev_cp = last_checkpoint.get(integral) or (str(output_file) + '.checkpoint')
                resume_from = prev_cp if Path(prev_cp).exists() else None
                new_checkpoint = str(new_output_file) + '.checkpoint'
                last_checkpoint[integral] = new_checkpoint

                if args.backend == 'local':
                    worker_script, argv, eff_cpus = build_worker_argv(
                        integral, new_output_file,
                        args.model_checkpoint,
                        args.straggler2_beam_width,  # wider beam
                        args.max_steps, args.prime,
                        topology_dir=args.topology,
                        paper_masters_only=args.paper_masters_only,
                        cpus=args.straggler2_cpus,   # more CPUs
                        beam_sort=args.beam_sort,
                        checkpoint_path=new_checkpoint,
                        checkpoint_interval=args.checkpoint_interval,
                        checkpoint_time_seconds=args.checkpoint_time_seconds,
                        resume_from=resume_from,
                        dedup_beam_by_content=True,  # force dedup at this level
                        use_delta_worker=args.use_delta_worker,
                        use_v6_worker=args.use_v6_worker,
                        use_v7_worker=args.use_v7_worker, v7_cpus=args.v7_cpus,
                    )
                    new_proc = launch_local_job(work_dir, job_name, worker_script, argv)
                    pending[integral] = (new_proc, new_output_file, time.time(), eff_cpus)
                    straggler2_integrals.add(integral)
                    total_jobs += 1
                    print(f"  STRAGGLER2: Re-launched I{list(integral)} (pid={new_proc.pid}) with "
                          f"{eff_cpus} CPUs, beam={args.straggler2_beam_width}, "
                          f"dedup ON (was running {job_runtime/3600:.1f}h)")
                    continue

                submit_file = create_condor_submit(
                    work_dir, integral, job_name, new_output_file,
                    args.model_checkpoint,
                    args.straggler2_beam_width,  # wider beam
                    args.max_steps, args.prime,
                    topology_dir=args.topology,
                    paper_masters_only=args.paper_masters_only,
                    cpus=args.straggler2_cpus,   # more CPUs
                    beam_sort=args.beam_sort,
                    checkpoint_path=new_checkpoint,
                    checkpoint_interval=args.checkpoint_interval,
                    checkpoint_time_seconds=args.checkpoint_time_seconds,
                    resume_from=resume_from,
                    dedup_beam_by_content=True,  # force dedup at this level
                    use_delta_worker=args.use_delta_worker,
                    use_v6_worker=args.use_v6_worker,
                use_v7_worker=args.use_v7_worker, v7_cpus=args.v7_cpus,
                    memory_gb=args.worker_memory_gb,
                )

                new_cluster_id = submit_condor_job(submit_file)
                if new_cluster_id:
                    pending[integral] = (new_cluster_id, new_output_file, time.time(), args.straggler2_cpus)
                    straggler2_integrals.add(integral)
                    total_jobs += 1
                    print(f"  STRAGGLER2: Resubmitted I{list(integral)} with "
                          f"{args.straggler2_cpus} CPUs, beam={args.straggler2_beam_width}, "
                          f"dedup ON (was running {job_runtime/3600:.1f}h)")
                else:
                    print(f"  WARNING: Failed to resubmit straggler2 I{list(integral)}")

        # Check for completed jobs
        completed_integrals = []
        for integral, (cluster_id, output_file, submit_time, cpus) in list(pending.items()):
            if output_file.exists():
                # Load result
                try:
                    with open(output_file, 'rb') as f:
                        result = pickle.load(f)

                    if result.get('success'):
                        # Cache the reduction: integral -> result expression
                        result_expr = result.get('final_expr', result.get('expr', {}))
                        cache[integral] = result_expr
                        steps = result.get('steps', 0)
                        total_steps += steps
                        worker_time = result.get('time', 0)
                        total_worker_time += worker_time

                        # Track critical path timing
                        d = depth.get(integral, 0)
                        if integral in parent:
                            ideal_finish[integral] = ideal_finish[parent[integral]] + worker_time
                        else:
                            ideal_finish[integral] = worker_time  # depth-0 root

                        # Track worker peak memory (raw and per-CPU)
                        worker_mem = result.get('peak_memory_kb', 0)
                        worker_mem_per_cpu = worker_mem // cpus
                        if worker_mem > max_worker_memory_kb:
                            max_worker_memory_kb = worker_mem
                        if worker_mem_per_cpu > max_worker_memory_per_cpu_kb:
                            max_worker_memory_per_cpu_kb = worker_mem_per_cpu

                        # Count new non-masters introduced
                        new_non_masters = get_non_masters(result_expr)
                        cached_count = len(new_non_masters & set(cache.keys()))
                        cache_hits += cached_count

                        # Propagate depth and parent to newly discovered children
                        for child_int in new_non_masters:
                            if child_int not in depth:
                                depth[child_int] = d + 1
                                parent[child_int] = integral

                        # Get sector and level info
                        sector = tuple(get_sector_mask(integral))
                        level = sum(sector)
                        w = weight(integral)

                        print(f"  Completed I{list(integral)} (L{level} w={w[:2]}): {len(result_expr)} terms, "
                              f"{steps} steps, {len(new_non_masters)} non-masters "
                              f"({cached_count} cached)")
                    else:
                        # Failed - cache as identity (no reduction possible)
                        cache[integral] = {integral: 1}
                        print(f"  Failed I{list(integral)} - caching as identity")

                    completed_integrals.append(integral)

                except Exception as e:
                    print(f"  Error loading result for I{list(integral)}: {e}")
                    # Don't cache - will retry

            elif (args.backend == 'local' and cluster_id is not None
                  and cluster_id.poll() is not None):
                # Local worker process exited but never wrote output_file (e.g.
                # uncaught exception, OOM-killed) -- no Condor hold/escalation
                # safety net exists for this locally, so detect it directly
                # rather than leaving the integral pending forever and the
                # CPU slot permanently stuck. Cache as identity, same as the
                # explicit success=False branch above.
                cache[integral] = {integral: 1}
                print(f"  Failed I{list(integral)} - local worker exited "
                      f"(code {cluster_id.returncode}) without producing "
                      f"output - caching as identity")
                completed_integrals.append(integral)

        # Remove completed from pending
        for integral in completed_integrals:
            del pending[integral]

        # Status update with hierarchical (level, r) breakdown.
        masters_count = sum(1 for i, c in expr.items() if c != 0 and is_master(i))
        non_masters_count = len(non_masters)

        # Frontier: max full_weight tuple still pending, and count at that tuple.
        if non_masters:
            frontier = max(full_weight(i) for i in non_masters)
            n_at_frontier = sum(1 for i in non_masters if full_weight(i) == frontier)
            work = work_units(non_masters)

            # Top-N (L, r) buckets — collapse s since it's secondary.
            Lr_counts = {}
            # Per-level: count of non-masters and max (r, s) weight.
            L_counts = {}
            L_maxw = {}
            for i in non_masters:
                L, r, s_ = full_weight(i)
                key = (L, r)
                Lr_counts[key] = Lr_counts.get(key, 0) + 1
                L_counts[L] = L_counts.get(L, 0) + 1
                prev = L_maxw.get(L)
                if prev is None or (r, s_) > prev:
                    L_maxw[L] = (r, s_)
            top = sorted(Lr_counts.items(), key=lambda kv: (-kv[0][0], -kv[0][1]))[:15]
            hist_str = " ".join(f"L{L}r{r}:{n}" for (L, r), n in top)
            # Per-level max-weight summary, highest level first.
            maxw_str = " ".join(
                f"L{L}: n={L_counts[L]} max=(r={L_maxw[L][0]},s={L_maxw[L][1]})"
                for L in sorted(L_counts, reverse=True)
            )

            # Rolling work-rate ETA over the last ~10 status prints.
            work_history.append((time.time(), work))
            if len(work_history) > 20:
                work_history.pop(0)
            eta_str = "ETA --"
            if len(work_history) >= 2:
                t0, w0 = work_history[0]
                dt = time.time() - t0
                dw = w0 - work
                if dt > 0 and dw > 0:
                    rate = dw / dt  # work units per second
                    eta_sec = work / rate if rate > 0 else 0
                    eta_str = f"-{dw/dt*60:.0f}/min, ETA {eta_sec/3600:.1f}h"

            print(f"[Iter {iteration}] {masters_count} masters, {non_masters_count} non-masters | "
                  f"frontier=(L={frontier[0]},r={frontier[1]},s={frontier[2]}) x {n_at_frontier} | "
                  f"work={work} ({eta_str}) | "
                  f"Pending: {len(pending)} | Cache: {len(cache)} | Hits: {cache_hits}")
            print(f"           [hist] {hist_str}")
            print(f"           [maxw] {maxw_str}")
        else:
            print(f"[Iter {iteration}] {masters_count} masters, 0 non-masters | "
                  f"Pending: {len(pending)} | Cache: {len(cache)} | Hits: {cache_hits}")

    # Final substitution
    expr = apply_substitutions(expr, cache, args.prime)

    # Final report
    elapsed = time.time() - start_time
    masters = {i: c for i, c in expr.items() if c != 0 and is_master(i)}
    non_masters = {i: c for i, c in expr.items() if c != 0 and not is_master(i)}

    # Get orchestrator peak memory
    orchestrator_peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    print()
    print('='*70)
    print('ASYNC REDUCTION COMPLETE')
    print('='*70)
    ideal_parallel_time = max(ideal_finish.values()) if ideal_finish else 0
    max_depth = max(depth.values()) if depth else 0

    print(f'Wall clock time: {elapsed:.1f}s ({elapsed/60:.1f} min)')
    print(f'Total worker runtime: {total_worker_time:.1f}s ({total_worker_time/60:.1f} min)')
    print(f'Ideal parallel time: {ideal_parallel_time:.1f}s ({ideal_parallel_time/60:.1f} min) '
          f'[critical path through {max_depth+1} depth levels]')
    print(f'Total jobs submitted: {total_jobs}')
    print(f'Stragglers resubmitted: {stragglers_resubmitted}')
    print(f'Total steps: {total_steps}')
    print(f'Cache size: {len(cache)}')
    print(f'Cache hits: {cache_hits}')
    print(f'Peak memory (orchestrator): {orchestrator_peak_kb/1024:.1f} MB ({orchestrator_peak_kb} KB)')
    print(f'Peak memory (max worker raw): {max_worker_memory_kb/1024:.1f} MB ({max_worker_memory_kb} KB)')
    print(f'Peak memory (max worker per-CPU): {max_worker_memory_per_cpu_kb/1024:.1f} MB ({max_worker_memory_per_cpu_kb} KB)')
    print()
    print(f'Final expression: {len(masters)} masters, {len(non_masters)} non-masters')

    if non_masters:
        print(f'\nWARNING: {len(non_masters)} non-masters remaining!')
        for integral, coeff in sorted(non_masters.items(), key=lambda x: (-weight(x[0])[0], x[0])):
            print(f'  I{list(integral)} coeff={coeff} weight={weight(integral)[:2]}')
    else:
        print('\nSUCCESS! All integrals reduced to masters.')

    print(f'\nFinal masters:')
    for integral, coeff in sorted(masters.items(), key=lambda x: (-weight(x[0])[0], x[0])):
        print(f'  I{list(integral)} coeff={coeff} weight={weight(integral)[:2]}')

    # Save result
    result = {
        # Replay-compatible header (for replay_reduction_path.py --orchestrator).
        # The per-worker IBP paths are NOT stored here -- they live in the
        # individual worker pickles under work_dir/results/ and are loaded
        # by the orchestrator-mode replay.
        'start_integral': starting_integral,
        'prime': args.prime,
        'final_expr': expr,
        'cache': cache,
        'total_jobs': total_jobs,
        'total_steps': total_steps,
        'elapsed_time': elapsed,
        'total_worker_time': total_worker_time,
        'cache_hits': cache_hits,
        'peak_memory_orchestrator_kb': orchestrator_peak_kb,
        'peak_memory_max_worker_kb': max_worker_memory_kb,
        'peak_memory_max_worker_per_cpu_kb': max_worker_memory_per_cpu_kb,
        'ideal_parallel_time': ideal_parallel_time,
    }
    with open(args.output, 'wb') as f:
        pickle.dump(result, f)

    print(f'\nResults saved to {args.output}')


if __name__ == '__main__':
    main()
