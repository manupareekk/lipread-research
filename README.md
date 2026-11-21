# speaker-adaptive-vsr

Visual-only lip-reading pipeline built on [Auto-AVSR](https://github.com/mpc001/auto_avsr). It takes a talking-head video, finds clean single-speaker sentences, runs the visual speech recognizer with no audio, scores it against a Whisper reference transcript, and renders a side-by-side visualization.

## What the model reads

1. MediaPipe face detection gives 4 keypoints per frame (both eyes, nose tip, mouth center).
2. Each frame is aligned to a mean face with a similarity transform, and a 96×96 patch is cut around the mouth.
3. Auto-AVSR's test transform center-crops that to **88×88, converts it to grayscale**, and normalizes it (mean 0.421, std 0.165).
4. The recognizer (3D-conv front end, Conformer encoder, CTC/attention decoder) reads only those raw pixels.

Landmarks are used only to align and crop. No lip points or outlines go into the model. `render.py` shows that exact 88×88 grayscale input.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
git clone https://github.com/mpc001/auto_avsr
# download a VSR checkpoint linked in the auto_avsr README, e.g. vsr_trlrs2lrs3vox2avsp_base.pth
```

## Pipeline

```bash
cd vsr
yt-dlp -f "bv*[height<=720][ext=mp4]" -o data/src.mp4 <url>        # or any local video
python transcribe.py data/src.mp4 data/words.jsonl                 # reference text (optional --windows 240:180 ...)
python select_segments.py data/src.mp4 data/words.jsonl data/segs.json
python cut_clips.py data/src.mp4 data/segs.json data/clips
python infer.py --auto-avsr ../auto_avsr --ckpt <ckpt.pth> --out data/hyps.jsonl data/clips/*.mp4
python score.py data/segs.json data/hyps.jsonl                     # per-sentence and corpus WER
python render.py data/segs.json data/hyps.jsonl out/demo.mp4 data/clips/<a>.mp4 data/clips/<b>.mp4
```

`render.py` expects Inter fonts (`Inter-Regular`, `Inter-SemiBold`, `InterDisplay-SemiBold`) in `--font-dir`.

## Measured results (untuned `vsr_lrs2lrs3vox2avsp` checkpoint, CPU)

| Speaker / source | Shot | Sentences | WER |
|---|---|---|---|
| Jensen Huang, CBS News interview | frontal close-up | 18 | ~22% |
| Mark Zuckerberg, 2005 interview | frontal close-up | 20 | 22% |
| Mark Zuckerberg, Sources podcast | three-quarter, medium | 35 | 34% |
| Alexandr Wang, David Senra podcast | three-quarter, wide | 39 | 45% |

Accuracy drops as the face turns away from the camera and gets smaller in the frame.

## Notes

- Source footage is not included. Respect the rights of whatever video you process.
- Whisper output is used only as the reference transcript for selection and scoring. The lip-reader never sees audio.
