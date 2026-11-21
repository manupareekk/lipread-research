import json, re, subprocess
import numpy as np
import cv2


def clip_name(seg):
    return f's{int(seg["s"] * 100):07d}.mp4'


def norm(text):
    return re.sub(r"[^a-z' ]", " ", text.lower().replace('-', ' ')).split()


def edit_distance(ref, hyp):
    d = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        prev, d[0] = d[:], i
        for j in range(1, len(hyp) + 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (ref[i - 1] != hyp[j - 1]))
    return d[-1]


def align(ref, hyp):
    """Return [(hyp_word, is_correct)] from a Levenshtein alignment."""
    n, m = len(ref), len(hyp)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    ok = [False] * m
    i, j = n, m
    while i > 0 and j > 0:
        if D[i][j] == D[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ok[j - 1] = ref[i - 1] == hyp[j - 1]
            i, j = i - 1, j - 1
        elif D[i][j] == D[i][j - 1] + 1:
            j -= 1
        else:
            i -= 1
    return list(zip(hyp, ok))


def read_video(path):
    cap, frames = cv2.VideoCapture(path), []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    return np.stack(frames)


def ffmpeg_frames(video, start, dur, fps=10, size=(640, 360)):
    w, h = size
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(start), '-t', str(dur), '-i', video,
                          '-vf', f'fps={fps},scale={w}:{h}', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                         capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def load_jsonl(path):
    return [json.loads(l) for l in open(path)]
