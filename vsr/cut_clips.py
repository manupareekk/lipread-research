"""Cut each selected segment to a 25 fps silent clip."""
import argparse, json, os, subprocess
from common import clip_name

ap = argparse.ArgumentParser()
ap.add_argument('video')
ap.add_argument('segments_json')
ap.add_argument('clip_dir')
a = ap.parse_args()
os.makedirs(a.clip_dir, exist_ok=True)
for d in json.load(open(a.segments_json)):
    f = os.path.join(a.clip_dir, clip_name(d))
    if not os.path.exists(f):
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', str(max(d['s'] - 0.1, 0)), '-t', str(d['e'] - d['s'] + 0.2),
                        '-i', a.video, '-vf', 'fps=25', '-an', '-c:v', 'libx264', '-crf', '12', f], check=True)
    print(f)
