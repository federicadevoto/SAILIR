"""
Meta-orchestrator for hexabox IBP reduction on SLAC S3DF (SLURM).

Adapted from reduction/meta_orchestrator.py (Condor) for SLURM/S3DF.

Submits one SLURM job per integral, monitors completion, and resubmits
automatically if no reduction.pkl is produced (up to MAX_RETRIES times).
Each retry uses --resume, so it continues from where the previous attempt
left off rather than starting from scratch.

Usage (from repo root on sdfiana019):
    screen -S meta
    cd /sdf/data/epptheory/federica/SAILIR
    python reduction/s3df/meta_orchestrator_s3df.py
    # or for a specific list (e.g. timed-out integrals):
    python reduction/s3df/meta_orchestrator_s3df.py --integral-list results/hexabox/timed_out.txt
    Ctrl-A D to detach; screen -r meta to re-attach

Config (edit the block below):
    MAX_ACTIVE   — max simultaneous SLURM jobs (8 x 32 CPUs = 256 = full allocation)
    MAX_RETRIES  — how many times to resubmit before giving up on an integral
    SLURM_TIME   — wall-clock limit per job (increase if integrals need more time)
    INTERVAL     — seconds between status polls
"""

import argparse
import subprocess
import time
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────────────
BASE     = Path('/sdf/data/epptheory/federica/SAILIR')
PY       = BASE / 'venv/bin/python'
MODEL    = BASE / 'checkpoints/hexabox_13M_biased/best_model.pt'
TOPOLOGY = BASE / 'topology_input/hexabox'
OUTBASE  = BASE / 'results/hexabox'

SLURM_ACCOUNT   = 'epptheory:qcd'
SLURM_PARTITION = 'milano'
SLURM_QOS       = 'normal'
SLURM_CPUS      = 32
SLURM_MEM       = '16G'
SLURM_TIME      = '3:00:00'   # per job; increase to e.g. 6:00:00 for harder integrals
JOB_NAME        = 'hb_meta'

MAX_ACTIVE  = 8    # 8 x 32 CPUs = 256 cores = full EPPTheory:QCD allocation
MAX_RETRIES = 3    # resubmit up to this many times before giving up
INTERVAL    = 60   # seconds between polls
# ── end config ────────────────────────────────────────────────────────────────


def log(m):
    print(f"[meta {time.strftime('%Y-%m-%d %H:%M:%S')}] {m}", flush=True)


def read_integrals(path):
    """Read integral list, deduplicate, skip blank lines and comments."""
    seen = set()
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line not in seen:
                seen.add(line)
                out.append(line)
    return out


def integral_outdir(integral):
    return OUTBASE / integral.replace(',', '_')


def is_done(integral):
    return (integral_outdir(integral) / 'reduction.pkl').exists()


def submit(integral):
    """Submit a SLURM job for this integral. Returns job ID string or None."""
    d = integral_outdir(integral)
    d.mkdir(parents=True, exist_ok=True)
    (d / 'logs').mkdir(exist_ok=True)
    (d / 'work').mkdir(exist_ok=True)

    # Remove stale timeout marker — meta orchestrator manages retries itself
    stale = d / 'reduction.timeout'
    if stale.exists():
        stale.unlink()

    py_cmd = (
        f"{PY} -u {BASE}/reduction/hierarchical_reduction.py"
        f" --topology {TOPOLOGY}"
        f" --integral={integral}"
        f" --output {d}/reduction.pkl"
        f" --work-dir {d}/work"
        f" --model-checkpoint {MODEL}"
        f" --beam_width 40 --max_steps 1000000 --prime 1009"
        f" --no-paper-masters-only --use-v7-worker --v7-cpus 1"
        f" --backend local --max-cpus {SLURM_CPUS}"
        f" --straggler-timeout 1000000000 --straggler2-timeout 1000000000"
        f" --check-interval 5 --max-concurrent 1000"
        f" --resume"
        f" 2>&1 | tee {d}/logs/hierarchical.log"
    )

    result = subprocess.run(
        ['sbatch', '--parsable',
         f'--job-name={JOB_NAME}',
         f'--account={SLURM_ACCOUNT}',
         f'--partition={SLURM_PARTITION}',
         f'--qos={SLURM_QOS}',
         '--ntasks=1',
         f'--cpus-per-task={SLURM_CPUS}',
         f'--mem={SLURM_MEM}',
         f'--time={SLURM_TIME}',
         f'--output={d}/logs/slurm_%j.out',
         f'--error={d}/logs/slurm_%j.err',
         '--wrap', f'source {BASE}/venv/bin/activate && {py_cmd}'],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        log(f"sbatch failed: {result.stderr.strip()}")
        return None
    # --parsable returns "jobid" or "jobid;cluster"
    return result.stdout.strip().split(';')[0]


def running_job_ids():
    """Return set of currently queued/running job IDs for our jobs, or None on error."""
    try:
        out = subprocess.run(
            ['squeue', '--me', f'--name={JOB_NAME}', '-h', '-o', '%i'],
            capture_output=True, text=True, timeout=30
        ).stdout
        return {line.strip() for line in out.splitlines() if line.strip()}
    except Exception as e:
        log(f"squeue error: {e}")
        return None


def main(integral_list_path):
    integrals = read_integrals(integral_list_path)
    log(f"Loaded {len(integrals)} unique integrals from {integral_list_path}")

    queue    = [i for i in integrals if not is_done(i)]
    n_done   = len(integrals) - len(queue)
    retries  = {}   # integral -> number of attempts so far
    active   = {}   # job_id   -> integral
    failed   = []

    log(f"Already done: {n_done} | To process: {len(queue)}")

    while queue or active:

        # ── check which active jobs finished ──────────────────────────────────
        running_ids = running_job_ids()
        if running_ids is None:
            time.sleep(INTERVAL)
            continue

        finished = {jid: integ for jid, integ in active.items()
                    if jid not in running_ids}
        for jid, integ in finished.items():
            del active[jid]
            if is_done(integ):
                log(f"SUCCESS {integ}  (job {jid})")
            else:
                attempt = retries.get(integ, 0) + 1
                retries[integ] = attempt
                if attempt <= MAX_RETRIES:
                    log(f"RETRY {attempt}/{MAX_RETRIES}: {integ}  (job {jid} — no pkl produced)")
                    queue.insert(0, integ)
                else:
                    log(f"GAVE UP ({MAX_RETRIES} retries exhausted): {integ}")
                    failed.append(integ)

        # ── submit new jobs while slots available ─────────────────────────────
        while queue and len(active) < MAX_ACTIVE:
            integ = queue.pop(0)
            jid = submit(integ)
            if jid:
                active[jid] = integ
                attempt = retries.get(integ, 0)
                log(f"SUBMITTED job {jid}: {integ}"
                    f"  (attempt {attempt+1}, active={len(active)}, queue={len(queue)})")
            else:
                queue.insert(0, integ)   # sbatch failed — try again next cycle
                break

        log(f"active={len(active)}  queue={len(queue)}  failed={len(failed)}")
        time.sleep(INTERVAL)

    n_succeeded = sum(1 for i in integrals if is_done(i)) - n_done
    log(f"ALL DONE — succeeded: {n_succeeded}  gave_up: {len(failed)}")
    if failed:
        log("Integrals that exhausted all retries:")
        for i in failed:
            log(f"  {i}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--integral-list',
        default=str(BASE / 'reduction/integrals_to_reduce.txt'),
        help='Path to integral list (one per line, duplicates ignored)'
    )
    args = parser.parse_args()
    main(args.integral_list)
