"""Fine-tune a per-speaker adapter (or the full model) on a few minutes of one speaker's video.

Non-full modes save only the trainable weights to <out>/adapter.pth plus <out>/meta.json, and keep
BatchNorm statistics and dropout frozen: in train mode a few minutes of data hurts WER badly.
"""
import argparse, json, os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import peft

p = argparse.ArgumentParser()
p.add_argument("--auto-avsr", required=True)
p.add_argument("--train-file", required=True); p.add_argument("--val-file", required=True)
p.add_argument("--out", required=True); p.add_argument("--epochs", type=int, default=15)
p.add_argument("--lr", type=float, default=5e-4); p.add_argument("--max-frames", type=int, default=1200)
p.add_argument("--mode", default="lora_enc"); p.add_argument("--r", type=int, default=8)
p.add_argument("--patience", type=int, default=3)
p.add_argument("--root", default="data/adapt", help="contains labels/ and clips/")
p.add_argument("--ckpt", required=True, help="e.g. vsr_trlrs2lrs3vox2avsp_base.pth")
a = p.parse_args()
a.root, a.out, a.ckpt = map(os.path.abspath, (a.root, a.out, a.ckpt))
sys.path.insert(0, a.auto_avsr); os.chdir(a.auto_avsr)

args = argparse.Namespace(modality="video", root_dir=a.root, train_file=a.train_file, val_file=a.val_file,
    test_file=a.val_file, pretrained_model_path=a.ckpt, transfer_frontend=False, transfer_encoder=False,
    warmup_epochs=1, max_epochs=a.epochs, max_frames=a.max_frames, lr=a.lr, weight_decay=0.03,
    ctc_weight=0.1, train_num_buckets=400, num_nodes=1, gpus=1, debug=False)
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from datamodule.data_module import DataModule
from lightning import ModelModule


def _train_step(self, batch, batch_idx):
    loss = self._step(batch, batch_idx, "train") / batch["inputs"].size(0)
    self.log("monitoring_step", torch.tensor(self.global_step, dtype=torch.float32))
    return loss


ModelModule.training_step = _train_step
if a.mode != "full":
    ModelModule.on_train_batch_start = lambda self, batch, batch_idx: peft.freeze_stats(self.model)
seed_everything(42, workers=True)
mm = ModelModule(args); dm = DataModule(args)
train_names = peft.apply(mm.model, a.mode, a.r)
n_tr = sum(p.numel() for p in mm.model.parameters() if p.requires_grad); n_all = sum(p.numel() for p in mm.model.parameters())
print("TRAINABLE", n_tr, n_all, flush=True)
ck = ModelCheckpoint(dirpath=a.out, monitor="loss_val", mode="min", save_top_k=1, filename="best")
tr = Trainer(max_epochs=a.epochs, devices=1, accelerator="gpu" if torch.cuda.is_available() else "cpu",
    precision="bf16-mixed" if torch.cuda.is_available() else 32, gradient_clip_val=10.0,
    callbacks=[ck, EarlyStopping(monitor="loss_val", mode="min", patience=a.patience)],
    logger=CSVLogger(a.out), reload_dataloaders_every_n_epochs=1, num_sanity_val_steps=0)
tr.fit(mm, dm)
best = torch.load(ck.best_model_path, map_location="cpu")["state_dict"]
sd = {k[len("model."):]: v for k, v in best.items() if k.startswith("model.")}
if a.mode in ("full", "full_eval"):
    torch.save(sd, f"{a.out}/model.pth")
else:
    torch.save({k: sd[k].to(torch.float32) for k in train_names}, f"{a.out}/adapter.pth")
json.dump(dict(mode=a.mode, r=a.r, trainable=n_tr, total=n_all, adapter_bytes=n_tr * 4), open(f"{a.out}/meta.json", "w"))
os.remove(ck.best_model_path)
print("BEST", float(ck.best_model_score))
