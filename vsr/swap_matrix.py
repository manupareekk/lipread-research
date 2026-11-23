"""Train per-speaker adapters across modes and data budgets, evaluate each on its own speaker's
held-out video, and evaluate every lora 15-minute adapter on every other speaker (the swap test)."""
import argparse, os, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("--auto-avsr", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--root", default="data/adapt")
ap.add_argument("--work", default="runs")
ap.add_argument("--eval-workers", type=int, default=2)
ap.add_argument("speakers", nargs="+")
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__)); PY = sys.executable
root, work = os.path.abspath(a.root), os.path.abspath(a.work)
for d in ("runs", "results", "logs"): os.makedirs(f"{work}/{d}", exist_ok=True)
LR = {"full": 5e-5, "full_eval": 5e-5, "lora": 5e-4, "lora_enc": 5e-4, "lora_dec": 5e-4, "bitfit": 5e-4, "frontend": 1e-4}
EP = {"2m": 30, "5m": 20, "15m": 12}
GRID = [("full", "2m")] + [(m, b) for m in ["lora", "bitfit", "full_eval", "frontend"] for b in ["2m", "5m", "15m"]] + [("lora_enc", "15m"), ("lora_dec", "15m")]
lock = threading.Lock()


def log(s):
    with lock: open(f"{work}/logs/times.txt", "a").write(s + "\n")


def run(cmd, logf):
    t = time.time(); rc = subprocess.call(cmd, stdout=open(logf, "w"), stderr=subprocess.STDOUT); return rc, int(time.time() - t)


def evaluate(spk_test, run_dir, out):
    if os.path.exists(out + ".done"): return
    if run_dir is None: extra = ["--ckpt", a.ckpt]
    elif os.path.exists(f"{run_dir}/model.pth"): extra = ["--ckpt", f"{run_dir}/model.pth"]
    elif os.path.exists(f"{run_dir}/adapter.pth"): extra = ["--ckpt", a.ckpt, "--adapter", run_dir]
    else: return
    rc, s = run([PY, f"{HERE}/eval_adapter.py", "--auto-avsr", a.auto_avsr, f"{root}/lists/{spk_test}_test.txt", out + ".jsonl"] + extra,
                f"{work}/logs/ev_{os.path.basename(out)}.txt")
    log(f"{os.path.basename(out)} eval_sec {s} rc {rc}")
    if rc == 0: open(out + ".done", "w").close()


pool = ThreadPoolExecutor(max_workers=a.eval_workers); futs = []
for spk in a.speakers:
    futs.append(pool.submit(evaluate, spk, None, f"{work}/results/{spk}_base"))
for mode, b in GRID:
    for spk in a.speakers:
        name = f"{spk}_{mode}_{b}"; d = f"{work}/runs/{name}"
        if not (os.path.exists(f"{d}/model.pth") or os.path.exists(f"{d}/adapter.pth")):
            rc, s = run([PY, f"{HERE}/train_adapter.py", "--auto-avsr", a.auto_avsr, "--ckpt", a.ckpt, "--root", root,
                         "--train-file", f"{spk}_train{b}.csv", "--val-file", f"{spk}_val.csv", "--out", d,
                         "--epochs", str(EP[b]), "--lr", str(LR[mode]), "--mode", mode], f"{work}/logs/ft_{name}.txt")
            log(f"{name} train_sec {s} rc {rc}")
        futs.append(pool.submit(evaluate, spk, d, f"{work}/results/{name}"))
        if mode == "lora" and b == "15m":
            for tgt in a.speakers:
                if tgt != spk: futs.append(pool.submit(evaluate, tgt, d, f"{work}/results/swap_{spk}_on_{tgt}"))
for f in futs: f.result()
log("ALLDONE")
