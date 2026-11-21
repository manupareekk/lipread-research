"""Per-segment and corpus word error rate of lip-reading hypotheses against the reference transcript."""
import argparse, json, os
from common import clip_name, edit_distance, load_jsonl, norm

ap = argparse.ArgumentParser()
ap.add_argument('segments_json')
ap.add_argument('hyps_jsonl')
a = ap.parse_args()
segs = {clip_name(d): d for d in json.load(open(a.segments_json))}
errs = words = 0
for h in load_jsonl(a.hyps_jsonl):
    s = segs[os.path.basename(h['clip'])]
    r, hy = norm(s['text']), norm(h['hyp'])
    e = edit_distance(r, hy)
    errs, words = errs + e, words + len(r)
    print(f"{e / len(r):.2f} | {s['text']} | {h['hyp'].lower()}")
print(f'WER {errs / words:.3f} over {words} words')
