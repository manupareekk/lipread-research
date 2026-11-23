"""Turn infer.py mouth crops into a per-speaker training list.

Writes <out>/clips/<speaker>/<clip>.mp4 (96x96 mouth crop, 25 fps) with the reference text next to
it as .txt, and appends "<path>\t<seconds>" lines to <out>/lists/<speaker>_<split>.txt.
"""
import argparse, json, os
import cv2, numpy as np
from common import clip_name

ap = argparse.ArgumentParser()
ap.add_argument('segments_json')
ap.add_argument('clips_dir', help='directory with infer.py *_mouth.npy files')
ap.add_argument('--speaker', required=True)
ap.add_argument('--split', required=True, choices=['train', 'val', 'test'])
ap.add_argument('--out', default='data/adapt')
a = ap.parse_args()

clips = os.path.join(a.out, 'clips', a.speaker); os.makedirs(clips, exist_ok=True)
os.makedirs(os.path.join(a.out, 'lists'), exist_ok=True)
with open(os.path.join(a.out, 'lists', f'{a.speaker}_{a.split}.txt'), 'a') as lst:
    for seg in json.load(open(a.segments_json)):
        npy = os.path.join(a.clips_dir, clip_name(seg).replace('.mp4', '_mouth.npy'))
        if not os.path.exists(npy):
            continue
        crop = np.load(npy)
        dst = os.path.abspath(os.path.join(clips, clip_name(seg)))
        w = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*'mp4v'), 25, (crop.shape[2], crop.shape[1]))
        for f in crop: w.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        w.release()
        open(dst[:-4] + '.txt', 'w').write(seg['text'])
        lst.write(f'{dst}\t{len(crop) / 25:.2f}\n')
