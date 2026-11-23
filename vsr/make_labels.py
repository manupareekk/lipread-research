"""Auto-AVSR label CSVs (train/val/test plus 2/5/15/30/60-minute train budgets) from speaker lists."""
import argparse, os, re, sys

ap = argparse.ArgumentParser()
ap.add_argument('--auto-avsr', required=True)
ap.add_argument('--root', default='data/adapt')
ap.add_argument('speakers', nargs='+')
a = ap.parse_args()
D = os.path.abspath(a.root)
sys.path.insert(0, a.auto_avsr); os.chdir(a.auto_avsr)
from datamodule.transforms import TextTransform

os.makedirs(f"{D}/labels", exist_ok=True)
tt = TextTransform()
norm = lambda s: " ".join(re.sub(r"[^A-Z0-9' ]", " ", re.sub(r"(\w) '(\w)", r"\1'\2", s).upper().replace("-", " ")).split())


def write(name, rows):
    with open(f"{D}/labels/{name}.csv", "w") as f:
        for p, dur in rows:
            txt = norm(open(p[:-4] + ".txt").read())
            if not txt: continue
            ids = " ".join(str(int(t)) for t in tt.tokenize(txt).tolist())
            f.write(f"clips,{os.path.relpath(p, D + '/clips')},{int(float(dur) * 25)},{ids}\n")


for spk in a.speakers:
    for split in ["train", "val", "test"]:
        rows = [l.rstrip("\n").split("\t") for l in open(f"{D}/lists/{spk}_{split}.txt") if l.strip()]
        write(f"{spk}_{split}", rows)
        if split == "train":
            for budget in [2, 5, 15, 30, 60]:
                acc, sub = 0, []
                for p, d in rows:
                    if acc >= budget * 60: break
                    sub.append((p, d)); acc += float(d)
                if acc >= budget * 60 * 0.9: write(f"{spk}_train{budget}m", sub)
print("ok")
