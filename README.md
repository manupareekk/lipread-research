# speaker-adaptive-vsr

Speaker-adaptive visual speech recognition (lip reading with no audio), built on [Auto-AVSR](https://github.com/mpc001/auto_avsr).

- **Pipeline:** take a talking-head video, find clean single-speaker sentences, run the visual speech recognizer, score it against a Whisper reference transcript, and render a side-by-side visualization.
- **Speaker adaptation:** train a small LoRA adapter on a few minutes of one person's video, evaluate on a held-out video of that person, and swap adapters between speakers to check that each one is speaker-specific.

## What the model reads

1. MediaPipe face detection gives 4 keypoints per frame (both eyes, nose tip, mouth center).
2. Each frame is aligned to a mean face with a similarity transform, and a 96×96 patch is cut around the mouth.
3. Auto-AVSR's test transform center-crops that to **88×88, converts it to grayscale**, and normalizes it (mean 0.421, std 0.165).
4. The recognizer (3D-conv front end, Conformer encoder, CTC/attention decoder) reads only those raw pixels.

Landmarks are used only to align and crop. No lip points or outlines go into the model. `render.py` shows that exact 88×88 grayscale input, the raw 0–255 values of an 8×8 lip patch, and the last 8 frames of the input sequence.

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

## Speaker adaptation

```bash
cd vsr
# per speaker: crops from infer.py -> 96x96 mouth clips + lists (use different source videos for train/val/test)
python export_crops.py data/segs.json data/clips --speaker alice --split train   # repeat for val, test
python make_labels.py --auto-avsr ../auto_avsr alice bob carol                    # labels + 2/5/15/30/60-min train budgets
python train_adapter.py --auto-avsr ../auto_avsr --ckpt <ckpt.pth> --mode lora_enc \
    --train-file alice_train15m.csv --val-file alice_val.csv --out runs/alice_lora_enc_15m
python eval_adapter.py --auto-avsr ../auto_avsr data/adapt/lists/alice_test.txt runs/alice.jsonl \
    --ckpt <ckpt.pth> --adapter runs/alice_lora_enc_15m
python swap_matrix.py --auto-avsr ../auto_avsr --ckpt <ckpt.pth> alice bob carol  # full grid + adapter swap test
python wer_table.py runs/results/*.jsonl
```

- `peft.py` implements the adaptation modes: `lora_enc`, `lora`, `lora_dec`, `bitfit`, `frontend`, `full`, `full_eval`.
- LoRA (r=8, alpha=16) wraps the q/k/v/out attention projections. `lora_enc` trains about 0.6M parameters (2.4 MB) and leaves the ~250M-parameter base model frozen.
- BatchNorm running stats and dropout stay frozen during adaptation. Training them on a few minutes of data makes WER much worse.
- Encoder-only LoRA carries the gain. Decoder-only LoRA does almost nothing.
- Training uses Auto-AVSR's data module, which reads clips with `torchvision.io.read_video`. That needs PyAV < 14.

## Zero-shot results (untuned `vsr_lrs2lrs3vox2avsp` checkpoint, CPU)

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
