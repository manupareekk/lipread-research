"""Side-by-side visualization: speaker (left), model input crop and word-by-word prediction (right).

Words the alignment marks as wrong are drawn red. The top-right panel shows the exact model input
(88x88 grayscale, nearest-neighbour upscaled), the raw 0-255 values of an 8x8 lip patch, and the
last 8 frames of the input sequence.
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
fN, fX = font('Inter-Regular', 12), font('Inter-Regular', 20)
Z, P0, C = 3, 40, 33  # crop zoom, lip patch origin, value cell size
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
        mk = min(int(si * len(mouth) / n), len(mouth) - 1)
        seq = [cv2.cvtColor(mouth[k][4:92, 4:92], cv2.COLOR_RGB2GRAY) for k in range(max(0, mk - 7), mk + 1)]
        seq = [seq[0]] * (8 - len(seq)) + seq
        cur = seq[-1]
        # model input crop, 3x nearest-neighbour
        fr.paste(Image.fromarray(cv2.resize(cur, (88 * Z, 88 * Z), interpolation=cv2.INTER_NEAREST)).convert('RGB'), (X, 130))
        # raw 0-255 values of an 8x8 patch on the lips
        GX, GY = X + 88 * Z + 56, 130
        d.rectangle((X + P0 * Z, 130 + P0 * Z, X + (P0 + 8) * Z - 1, 130 + (P0 + 8) * Z - 1), outline=(235, 235, 235))
        d.line((X + (P0 + 8) * Z, 130 + P0 * Z, GX, GY), fill=(90, 90, 100))
        d.line((X + (P0 + 8) * Z, 130 + (P0 + 8) * Z - 1, GX, GY + 8 * C), fill=(90, 90, 100))
        for r in range(8):
            for q in range(8):
                v, x0, y0 = int(cur[P0 + r, P0 + q]), GX + q * C, GY + r * C
                d.rectangle((x0, y0, x0 + C - 1, y0 + C - 1), fill=(v, v, v))
                tw = d.textlength(str(v), font=fN)
                d.text((x0 + (C - tw) / 2, y0 + 9), str(v), font=fN, fill=(0, 0, 0) if v > 110 else (225, 225, 225))
        d.text((X, 130 + 88 * Z + 10), 'model input: 88\u00d788 grayscale', font=fX, fill=GRY)
        d.text((GX, GY + 8 * C + 10), 'raw pixel values', font=fX, fill=GRY)
        # last 8 frames of the input sequence
        SY, TW = 440, 80
        for k, im in enumerate(seq):
            fr.paste(Image.fromarray(cv2.resize(im, (TW, TW), interpolation=cv2.INTER_AREA)).convert('RGB'), (X + k * (TW + 8), SY))
        d.rectangle((X + 7 * (TW + 8) - 2, SY - 2, X + 7 * (TW + 8) + TW + 1, SY + TW + 1), outline=(235, 235, 235))
        d.text((X, SY + TW + 8), f'input sequence, {FPS} frames/s', font=fX, fill=GRY)
        p = min(1, (t + 1) / (0.8 * N))
        yy = 590
        d.text((X, yy), 'Lip-reader (no audio)', font=fL, fill=GRY)
        yy = wrap(d, hy, fC, X, yy + 40, 740, round(p * len(hy)), lambda ok: GRN if ok else RED) + 30
        if p >= 1:
            d.text((X, yy), 'Reference transcript', font=fL, fill=GRY)
            wrap(d, ref_words, fR, X, yy + 40, 740, len(ref_words), lambda _: GRY, lh=46)
        enc.stdin.write(fr.tobytes())
enc.stdin.close()
enc.wait()
