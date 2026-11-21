"""Visual-only speech recognition with Auto-AVSR.

The model input is the aligned 96x96 mouth crop, center-cropped to 88x88 and converted to grayscale
(Auto-AVSR VideoTransform 'test'). Face landmarks are used only to align and cut the crop; no
landmark/outline information is given to the recognizer.
"""
import argparse, json, os, sys
import numpy as np, torch
from common import read_video

ap = argparse.ArgumentParser()
ap.add_argument('--auto-avsr', required=True, help='path to a clone of github.com/mpc001/auto_avsr')
ap.add_argument('--ckpt', required=True, help='e.g. vsr_trlrs2lrs3vox2avsp_base.pth / vsr_lrs2lrs3vox2avsp.pth')
ap.add_argument('--out', required=True, help='hypotheses jsonl (appended)')
ap.add_argument('--threads', type=int, default=6)
ap.add_argument('clips', nargs='+')
a = ap.parse_args()

clips = [os.path.abspath(c) for c in a.clips]
out_path, ckpt = os.path.abspath(a.out), os.path.abspath(a.ckpt)
sys.path.insert(0, a.auto_avsr)
os.chdir(a.auto_avsr)
from lightning import ModelModule
from datamodule.transforms import VideoTransform
from preparation.detectors.mediapipe.detector import LandmarksDetector
from preparation.detectors.mediapipe.video_process import VideoProcess

torch.set_num_threads(a.threads)
detector, cropper, transform = LandmarksDetector(), VideoProcess(convert_gray=False), VideoTransform(subset='test')
model = ModelModule(argparse.Namespace(modality='video'))
model.model.load_state_dict(torch.load(ckpt, map_location='cpu'))
model.eval()

with open(out_path, 'a') as out:
    for f in clips:
        video = read_video(f)
        crop = cropper(video, detector(video))
        if crop is None:
            print('no face', f)
            continue
        np.save(f.replace('.mp4', '_mouth.npy'), crop.astype(np.uint8))
        with torch.no_grad():
            hyp = model(transform(torch.tensor(crop).permute(0, 3, 1, 2)))
        print(f, hyp, flush=True)
        out.write(json.dumps({'clip': os.path.basename(f), 'hyp': hyp}) + '\n')
        out.flush()
