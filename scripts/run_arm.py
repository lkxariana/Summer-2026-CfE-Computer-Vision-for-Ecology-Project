#!/usr/bin/env python3
"""Run one arm (a model configuration) through the evaluation protocol and skip whatever already has a bundle.

  scripts/run_arm.py configs/arms/final.json                       # 4 regimes x 3 seeds + within-site x 3 seeds, GPU 0
  scripts/run_arm.py configs/arms/final.json --splits cold_plant --seeds 42 --no-local --gpu 1
  scripts/run_arm.py configs/arms/ablations/*.json --seeds 42       # every ablation, seed 42 only
  scripts/run_arm.py configs/arms/final.json --part test            # the test split (run once, at the end)

An arm file names the bundle, the retriever (REGISTRY key + kwargs) and optionally a re-ranker; see configs/arms/README.md.
Each (split, seed) is a separate process (python -m antheia.eval.ladder | rerank | localnets | localnets_rerank), so GPU
memory is released between runs and a crash in one run does not stop the others. Output lines with metrics or errors are
echoed and appended to logs/<name>.log.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from antheia.paths import FIELD_DIR, RUNS  # noqa: E402

PY = sys.executable
SPLITS = ["cold_plant", "cold_poll", "cold_both", "warm"]
SEEDS = [42, 0, 1]


def have_bundle(name, split, part, seed):
    return any(RUNS.glob(f"{name}/*/{split}/{part}/s{seed}/metrics.json"))


def have_local(name, seed):
    return any(RUNS.glob(f"{name}/*/localnet/s{seed}/metrics.json"))


def run(cmd, gpu, log, dry):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(REPO / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    print("  $", " ".join(c if " " not in c else repr(c) for c in cmd[2:6]), "...", flush=True)
    if dry:
        return 0
    with open(log, "a") as fh:
        fh.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] {' '.join(cmd)}\n")
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            fh.write(line)
            if any(k in line for k in ("AUPR", "networks", "recall@", "Traceback", "Error", "excluded")):
                print("   ", line.rstrip(), flush=True)
        proc.wait()
    return proc.returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="+", help="arm json file(s)")
    ap.add_argument("--splits", nargs="*", default=SPLITS, choices=SPLITS + [])
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--part", default="val", choices=["val", "test"])
    ap.add_argument("--no-local", action="store_true", help="skip the within-site evaluation")
    ap.add_argument("--only-local", action="store_true")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    (REPO / "logs").mkdir(exist_ok=True)
    failures = []
    for arm_path in args.arms:
        arm = json.load(open(arm_path))
        name, rmodel = arm["name"], arm["retriever_model"]
        rcfg = dict(arm.get("retriever", {})); rcfg.setdefault("field_dir", str(FIELD_DIR))
        rr = arm.get("reranker"); log = REPO / "logs" / f"{name}.log"
        if rr is not None:
            rr = dict(rr); rr.setdefault("field_dir", str(FIELD_DIR))
        print(f"== {name}: {arm.get('description', '')}", flush=True)
        if not args.only_local:
            for split in args.splits:
                for seed in args.seeds:
                    if have_bundle(name, split, args.part, seed):
                        print(f"  [have] {split}/{args.part} s{seed}"); continue
                    if rr is None:
                        cmd = [PY, "-m", "antheia.eval.ladder", "--model", rmodel, "--name", name, "--config", json.dumps(rcfg),
                               "--split", split, "--part", args.part, "--seed", str(seed)]
                    else:
                        cmd = [PY, "-m", "antheia.eval.rerank", "--name", name, "--retriever-model", rmodel, "--retriever-config", json.dumps(rcfg),
                               "--config", json.dumps(rr), "--split", split, "--part", args.part, "--seed", str(seed)]
                    print(f"  [run ] {split}/{args.part} s{seed}", flush=True)
                    if run(cmd, args.gpu, log, args.dry_run):
                        failures.append((name, split, seed))
        if not args.no_local and args.part == "val":
            lcfg = dict(rcfg, **arm.get("within_site_retriever_overrides", {}))
            lname = arm.get("within_site_name", name)
            for seed in args.seeds:
                if have_local(lname, seed):
                    print(f"  [have] within-site s{seed}"); continue
                if rr is None:
                    cmd = [PY, "-m", "antheia.eval.localnets", "--model", rmodel, "--name", lname, "--config", json.dumps(lcfg), "--seed", str(seed)]
                else:
                    cmd = [PY, "-m", "antheia.eval.localnets_rerank", "--name", lname, "--retriever-model", rmodel, "--retriever-config", json.dumps(lcfg),
                           "--config", json.dumps(rr), "--seed", str(seed)]
                print(f"  [run ] within-site s{seed}", flush=True)
                if run(cmd, args.gpu, log, args.dry_run):
                    failures.append((lname, "localnet", seed))
    if failures:
        print("FAILED:", failures); sys.exit(1)
    print("done")


if __name__ == "__main__":
    main()
