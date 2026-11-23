"""Decode a speaker's test list with the base model, an adapter (--adapter) or a full fine-tune (--ckpt)."""
import argparse, json, os, re, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import peft
from common import read_video

p = argparse.ArgumentParser()
p.add_argument("--auto-avsr", required=True)
p.add_argument("list", help="<speaker>_test.txt: '<mouth crop mp4>\\t<seconds>' per line")
p.add_argument("out", help="jsonl (resumable)")
p.add_argument("--ckpt", required=True)
p.add_argument("--adapter", default=None, help="train_adapter.py output dir")
p.add_argument("--shard", default="0/1")
a = p.parse_args()
a.list, a.out, a.ckpt = map(os.path.abspath, (a.list, a.out, a.ckpt))
if a.adapter: a.adapter = os.path.abspath(a.adapter)
sys.path.insert(0, a.auto_avsr); os.chdir(a.auto_avsr)
from datamodule.transforms import VideoTransform
from lightning import ModelModule

k, n = map(int, a.shard.split("/"))
torch.set_num_threads(int(os.environ.get("THREADS", "8")))
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = ModelModule(argparse.Namespace(modality="video", pretrained_model_path=a.ckpt))
if a.adapter:
    meta = json.load(open(os.path.join(a.adapter, "meta.json"))); peft.apply(m.model, meta["mode"], meta["r"])
    missing, unexpected = m.model.load_state_dict(torch.load(os.path.join(a.adapter, "adapter.pth"), map_location="cpu"), strict=False)
    assert not unexpected, unexpected
m = m.eval().to(dev)
tf = VideoTransform("test")
norm = lambda s: " ".join(re.sub(r"[^a-z0-9' ]", " ", s.lower().replace("-", " ")).split())
clips = [l.split("\t")[0].strip() for l in open(a.list) if l.strip()][k::n]
done = {json.loads(l)["clip"] for l in open(a.out)} if os.path.exists(a.out) else set()
with open(a.out, "a") as f:
    for c in clips:
        if c in done: continue
        t = time.time()
        v = torch.tensor(read_video(c)).permute(0, 3, 1, 2)
        with torch.no_grad(): hyp = m(tf(v.float()).to(dev))
        ref = open(c[:-4] + ".txt").read()
        f.write(json.dumps(dict(clip=c, ref=norm(ref), hyp=norm(hyp), sec=round(time.time() - t, 2))) + "\n"); f.flush()
