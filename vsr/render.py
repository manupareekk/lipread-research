"""Side-by-side visualization: speaker (left), model input crop and word-by-word prediction (right).

Words the alignment marks as wrong are drawn red. The mouth panel shows the exact model input
(88x88 grayscale), upscaled with nearest-neighbour.
"""
import argparse, json, os, subprocess
import numpy as np, cv2, mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
from common import align, clip_name, load_jsonl, norm, read_video

ap = argparse.ArgumentParser()
ap.add_argument('segments_json')
ap.add_argument('hyps_jsonl')
ap.add_argument('out_mp4')
ap.add_argument('clips', nargs='+')
ap.add_argument('--speed', type=float, default=0.8)
ap.add_argument('--font-dir', default=os.path.expanduser('~/.fonts'))
a = ap.parse_args()

W, H, FPS, SP = 1920, 1080, 25, a.speed
font = lambda n, s: ImageFont.truetype(os.path.join(a.font_dir, f'{n}.ttf'), s)
fT, fL, fC, fS, fR = font('InterDisplay-SemiBold', 40), font('Inter-SemiBold', 26), font('InterDisplay-SemiBold', 46), \
    font('Inter-Regular', 26), font('Inter-Regular', 34)
RED, GRN, WHT, GRY = (255, 95, 95), (90, 225, 140), (240, 240, 240), (150, 150, 160)
segs = {clip_name(d): d for d in json.load(open(a.segments_json))}
hyps = {os.path.basename(h['clip']): h['hyp'] for h in load_jsonl(a.hyps_jsonl)}
fd = mp.solutions.face_detection.FaceDetection(model_selection=1)


def wrap(d, words, fnt, x0, y, maxw, shown, col, lh=60):
    x = x0
    for w, c in words[:shown]:
        tw = d.textlength(w + ' ', font=fnt)
        if x + tw > x0 + maxw:
            x, y = x0, y + lh
        d.text((x, y), w, font=fnt, fill=col(c))
        x += tw
    return y + lh


enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS),
                        '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', '-movflags', '+faststart', a.out_mp4],
                       stdin=subprocess.PIPE)
for path in a.clips:
    key = os.path.basename(path)
    s, full = segs[key], read_video(path)
    n, fh, fw = len(full), full.shape[1], full.shape[2]
    box = fd.process(full[n // 2]).detections[0].location_data.relative_bounding_box
    cx, cy, c = (box.xmin + box.width / 2) * fw, (box.ymin + box.height / 2) * fh, int(min(fh, fw) * 0.83)
    x = int(min(max(cx - c / 2, 0), fw - c))
    y = int(min(max(cy - c * 0.42, 0), fh - c))
    face = [cv2.resize(f[y:y + c, x:x + c], (1080, 1080), interpolation=cv2.INTER_LANCZOS4) for f in full]
    mouth = np.load(path.replace('.mp4', '_mouth.npy'))
    hy = align(norm(s['text']), norm(hyps[key]))
    ref_words = [(w, 0) for w in s['text'].split()]
    N = round(n / SP)
    for t in range(N):
        si = min(int(t * SP), n - 1)
        fr = Image.new('RGB', (W, H), (14, 14, 18))
        fr.paste(Image.fromarray(face[si]), (0, 0))
        d, X = ImageDraw.Draw(fr), 1140
        d.rounded_rectangle((24, 24, 222, 78), 12, fill=(0, 0, 0)); d.text((44, 34), 'SOUND OFF', font=fL, fill=WHT)
        d.rounded_rectangle((234, 24, 412, 78), 12, fill=(0, 0, 0)); d.text((254, 34), f'{SP:g}\u00d7 speed', font=fL, fill=WHT)
        d.text((X, 60), 'Reading lips only', font=fT, fill=WHT)
        m = mouth[min(int(si * len(mouth) / n), len(mouth) - 1)][4:92, 4:92]
        fr.paste(Image.fromarray(cv2.resize(cv2.cvtColor(m, cv2.COLOR_RGB2GRAY), (300, 300),
                                            interpolation=cv2.INTER_NEAREST)).convert('RGB'), (X, 130))
        d.text((X + 320, 150), 'what the model sees', font=fS, fill=GRY)
        d.text((X + 320, 186), '(88\u00d788 grayscale crop)', font=fS, fill=GRY)
        p = min(1, (t + 1) / (0.8 * N))
        yy = 480
        d.text((X, yy), 'Lip-reader (no audio)', font=fL, fill=GRY)
        yy = wrap(d, hy, fC, X, yy + 40, 740, round(p * len(hy)), lambda ok: GRN if ok else RED) + 30
        if p >= 1:
            d.text((X, yy), 'Reference transcript', font=fL, fill=GRY)
            wrap(d, ref_words, fR, X, yy + 40, 740, len(ref_words), lambda _: GRY, lh=46)
        enc.stdin.write(fr.tobytes())
enc.stdin.close()
enc.wait()
