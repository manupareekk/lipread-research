"""Word-timestamped reference transcript with faster-whisper (used only for ground truth / segmentation)."""
import argparse, json, subprocess, tempfile, os
from faster_whisper import WhisperModel

ap = argparse.ArgumentParser()
ap.add_argument('video')
ap.add_argument('out_jsonl')
ap.add_argument('--windows', nargs='*', default=[], help='start:duration windows in seconds (default: whole file)')
ap.add_argument('--model', default='medium.en')
ap.add_argument('--threads', type=int, default=8)
a = ap.parse_args()

m = WhisperModel(a.model, device='cpu', compute_type='int8', cpu_threads=a.threads)
windows = [tuple(map(float, w.split(':'))) for w in a.windows] or [(0.0, None)]
with open(a.out_jsonl, 'w') as f, tempfile.TemporaryDirectory() as td:
    for start, dur in windows:
        wav = os.path.join(td, 'a.wav')
        cmd = ['ffmpeg', '-v', 'error', '-y', '-ss', str(start)] + (['-t', str(dur)] if dur else []) + \
              ['-i', a.video, '-ac', '1', '-ar', '16000', wav]
        subprocess.run(cmd, check=True)
        segs, _ = m.transcribe(wav, word_timestamps=True, vad_filter=True, beam_size=5)
        for s in segs:
            f.write(json.dumps({'s': s.start + start, 'e': s.end + start, 'text': s.text,
                                'words': [[w.start + start, w.end + start, w.word] for w in s.words]}) + '\n')
            f.flush()
