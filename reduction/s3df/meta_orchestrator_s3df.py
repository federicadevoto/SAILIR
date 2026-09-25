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
from collections import Counter
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
SLURM_TIME      = '3:00:00'   # per job; increase to e.g. 6:00:00 for harder integrals
JOB_NAME        = 'hb_meta'

# Memory escalation ladder. A job killed with OUT_OF_MEMORY is retried one rung
# higher rather than repeatedly dying at the same size. 100G is roughly the
# proportional share of a 480G/128-core milan node for 32 CPUs; the original
# 16G caused mass OOM kills (each beam worker peaks ~1.5G, many resident).
MEM_LADDER = ['100G', '200G', '400G']

# Failure states caused by the environment rather than by the integral being
# hard. These earn a fresh attempt WITHOUT consuming the MAX_RETRIES budget —
# otherwise an OOM or a bad node silently burns the retries an integral needs.
INFRA_STATES = {'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED',
                'CANCELLED', 'BOOT_FAIL', 'UNKNOWN'}

MAX_ACTIVE   = 8   # 8 x 32 CPUs = 256 cores = full EPPTheory:QCD allocation
MAX_RETRIES  = 3   # genuine (non-infra) retries before giving up on an integral
MAX_ATTEMPTS = 8   # hard cap on total submissions, infra retries included
INTERVAL     = 60  # seconds between polls
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


def submit(integral, mem):
    """Submit a SLURM job for this integral at the given memory size.
    Returns job ID string or None."""
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
         f'--mem={mem}',
         f'--time={SLURM_TIME}',
         f'--output={d}/logs/slurm_%j.out',
         f'--error={d}/logs/slurm_%j.err',
         '--wrap', f'source {BASE}/venv/bin/activate && {py_cmd}'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )
    if result.returncode != 0:
        log(f"sbatch failed: {result.stderr.strip()}")
        return None
    # --parsable returns "jobid" or "jobid;cluster"
    return result.stdout.strip().split(';')[0]


def queued_jobs():
    """Return {job_id: reason} for our queued/running jobs, or None on error."""
    try:
        out = subprocess.run(
            ['squeue', '--me', f'--name={JOB_NAME}', '-h', '-o', '%i|%R'],
            stdout=subprocess.PIPE, universal_newlines=True, timeout=30
        ).stdout
        jobs = {}
        for line in out.splitlines():
            if '|' not in line:
                continue
            jid, reason = line.split('|', 1)
            jobs[jid.strip()] = reason.strip()
        return jobs
    except Exception as e:
        log(f"squeue error: {e}")
        return None


def job_state(jid):
    """Final accounting state of a finished job: OUT_OF_MEMORY, TIMEOUT,
    COMPLETED, NODE_FAIL, ... Returns 'UNKNOWN' if sacct has no record yet
    (accounting can lag a little behind the job leaving squeue)."""
    try:
        out = subprocess.run(
            ['sacct', '-j', jid, '--format=State', '-X', '--parsable2', '--noheader'],
            stdout=subprocess.PIPE, universal_newlines=True, timeout=30).stdout
    except Exception as e:
        log(f"sacct error for {jid}: {e}")
        return 'UNKNOWN'
    for line in out.splitlines():
        s = line.strip()
        if s:
            return s.split()[0]      # "CANCELLED by 1234" -> "CANCELLED"
    return 'UNKNOWN'


def is_held(reason):
    """True for a job SLURM has parked and will never start on its own.

    'launch failed requeued held' (bad node / prolog failure), plus admin and
    user holds. Held jobs sit in squeue forever, so without this they would
    occupy an active slot for the rest of the run."""
    return 'held' in reason.lower()


def main(integral_list_path):
    integrals = read_integrals(integral_list_path)
    log(f"Loaded {len(integrals)} unique integrals from {integral_list_path}")

    queue    = [i for i in integrals if not is_done(i)]
    n_done   = len(integrals) - len(queue)
    retries  = {}              # integral -> genuine (non-infra) retries used
    attempts = {}              # integral -> total submissions, infra included
    mem_rung = {}              # integral -> index into MEM_LADDER
    states   = Counter()       # final SLURM state -> count, for the summary
    active   = {}              # job_id   -> integral
    failed   = []

    log(f"Already done: {n_done} | To process: {len(queue)}")

    while queue or active:

        # ── check which active jobs finished ──────────────────────────────────
        jobs = queued_jobs()
        if jobs is None:
            time.sleep(INTERVAL)
            continue

        # Cancel held jobs (failed launch, admin/user hold). They never run and
        # would otherwise hold an active slot forever; dropping them here makes
        # them fall through to the finished/retry path below.
        for jid, reason in list(jobs.items()):
            if is_held(reason):
                log(f"HELD job {jid} ({reason}) — cancelling so it can be retried")
                subprocess.run(['scancel', jid],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                del jobs[jid]

        finished = {jid: integ for jid, integ in active.items()
                    if jid not in jobs}
        for jid, integ in finished.items():
            del active[jid]
            if is_done(integ):
                log(f"SUCCESS {integ}  (job {jid})")
                continue

            state = job_state(jid)
            states[state] += 1
            attempts[integ] = attempts.get(integ, 0) + 1

            if state == 'OUT_OF_MEMORY':
                # Resource failure, not a hard integral: step up the memory and
                # retry for free. Only genuine failures spend MAX_RETRIES.
                rung = mem_rung.get(integ, 0)
                if rung + 1 < len(MEM_LADDER):
                    mem_rung[integ] = rung + 1
                    log(f"OOM {integ} at {MEM_LADDER[rung]} (job {jid}) "
                        f"— retrying at {MEM_LADDER[rung+1]}")
                else:
                    log(f"OOM {integ} at {MEM_LADDER[rung]} (job {jid}) "
                        f"— already at largest rung, retrying")
            elif state in INFRA_STATES:
                log(f"{state} {integ} (job {jid}) — infra failure, "
                    f"retrying without spending a retry")
            else:
                retries[integ] = retries.get(integ, 0) + 1
                log(f"{state} {integ} (job {jid}) "
                    f"— retry {retries[integ]}/{MAX_RETRIES}")

            if retries.get(integ, 0) > MAX_RETRIES:
                log(f"GAVE UP ({MAX_RETRIES} real retries exhausted): {integ}")
                failed.append(integ)
            elif attempts[integ] >= MAX_ATTEMPTS:
                log(f"GAVE UP ({MAX_ATTEMPTS} attempts incl. infra failures): {integ}")
                failed.append(integ)
            else:
                queue.insert(0, integ)

        # ── submit new jobs while slots available ─────────────────────────────
        while queue and len(active) < MAX_ACTIVE:
            integ = queue.pop(0)
            mem = MEM_LADDER[mem_rung.get(integ, 0)]
            jid = submit(integ, mem)
            if jid:
                active[jid] = integ
                log(f"SUBMITTED job {jid}: {integ}"
                    f"  (attempt {attempts.get(integ, 0)+1}, mem={mem}, "
                    f"active={len(active)}, queue={len(queue)})")
            else:
                queue.insert(0, integ)   # sbatch failed — try again next cycle
                break

        log(f"active={len(active)}  queue={len(queue)}  failed={len(failed)}")
        time.sleep(INTERVAL)

    n_succeeded = sum(1 for i in integrals if is_done(i)) - n_done
    log(f"ALL DONE — succeeded: {n_succeeded}  gave_up: {len(failed)}")
    if states:
        log("Job outcomes seen (a failure here is per-job, not per-integral):")
        for state, n in states.most_common():
            log(f"  {state:<16} {n}")
    if failed:
        log("Integrals that exhausted all retries:")
        for i in failed:
            log(f"  {i}  (attempts={attempts.get(i, 0)}, "
                f"real_retries={retries.get(i, 0)}, "
                f"max_mem={MEM_LADDER[mem_rung.get(i, 0)]})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--integral-list',
        default=str(BASE / 'reduction/integrals_to_reduce.txt'),
        help='Path to integral list (one per line, duplicates ignored)'
    )
    args = parser.parse_args()
    main(args.integral_list)
