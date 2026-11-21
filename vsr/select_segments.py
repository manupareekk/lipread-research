"""Keep sentence segments with no shot cut and a single, large enough face."""
import argparse, json
import numpy as np, mediapipe as mp
from common import ffmpeg_frames, load_jsonl

ap = argparse.ArgumentParser()
ap.add_argument('video')
ap.add_argument('transcript_jsonl')
ap.add_argument('out_json')
ap.add_argument('--min-dur', type=float, default=2.0)
ap.add_argument('--max-dur', type=float, default=6.0)
ap.add_argument('--min-words', type=int, default=5)
ap.add_argument('--min-face-width', type=float, default=0.08, help='face width as a fraction of frame width')
ap.add_argument('--max-cut-diff', type=float, default=20.0)
a = ap.parse_args()

fd = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6)
out = []
for d in load_jsonl(a.transcript_jsonl):
    dur, txt = d['e'] - d['s'], d['text'].strip()
    if not (a.min_dur <= dur <= a.max_dur) or len(txt.split()) < a.min_words:
        continue
    fr = ffmpeg_frames(a.video, max(d['s'] - 0.1, 0), dur + 0.2)
    if len(fr) < 2 or np.abs(np.diff(fr[:, ::4, ::4].astype(int), axis=0)).mean((1, 2, 3)).max() > a.max_cut_diff:
        continue
    widths = []
    for f in fr[::3]:
        r = fd.process(f).detections
        widths.append(r[0].location_data.relative_bounding_box.width if r and len(r) == 1 else 0)
    if min(widths) < a.min_face_width:
        continue
    out.append(dict(s=d['s'], e=d['e'], text=txt, face_width=float(np.mean(widths))))
json.dump(out, open(a.out_json, 'w'), indent=0)
print(len(out), 'segments')
for o in out:
    print(f"{o['s']:8.1f} {o['face_width']:.2f} {o['text']}")
