# infer.py
import argparse, os, time
import torch
import yaml
from pathlib import Path
import numpy as np
from PIL import Image
from diffusion import Diffusion
from networks import Model
from lhc_dataset import ChannelDataset
from utils import dict2namespace
from datetime import datetime

def load_config(config_yaml, device):
    with open(config_yaml, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = dict2namespace(cfg)
    cfg.device = device
    return cfg

def tensor_to_image(t):
    # t assumed to be (C,H,W) in [-1,1] or [0,1] - we handle both
    t = t.detach().cpu().numpy()
    if t.ndim == 3 and t.shape[0] == 1:
        im = t[0]
    elif t.ndim == 3:
        im = t.transpose(1,2,0)
    else:
        im = t
    # if values appear in [-1,1], convert to [0,1]
    if im.min() < -0.1:
        im = (im + 1.0) * 0.5
    im = np.clip(im, 0.0, 1.0)
    im = (im * 255).astype(np.uint8)
    if im.ndim == 2:
        return Image.fromarray(im, mode='L')
    else:
        return Image.fromarray(im)

def prepare_output_path(out_arg, model_type):
    out_path = Path(out_arg)
    if out_path.is_dir() or str(out_arg).endswith(os.sep):
        # directory provided -> choose a filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"recon_{model_type}_{timestamp}.png"
        out_path = out_path / fname
    else:
        # if parent dir doesn't exist, create it
        if not out_path.parent.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--run_dir', type=str, default='.', help='Root containing saved_models/ (or project root).')
    p.add_argument('--model_type', type=str, default='best', choices=['best','epoch'], help='Which model to use: "best" or "epoch"')
    p.add_argument('--epoch', type=int, default=None, help='Epoch number if model_type==epoch')
    p.add_argument('--lhc', default=-1, type=int, help='If >=0 use LHC dataset settings (set in training).')
    p.add_argument('--device', default=0, type=int)
    p.add_argument('--seed', default=0, type=int)
    p.add_argument('--steps', default=1000, type=int, help='Number of sampling steps for ddpm')
    p.add_argument('--out', default='recon.png', help='Output filename or directory (PNG). If directory, a filename is chosen automatically.')
    args = p.parse_args()

    # Candidate saved_models locations
    
    supplied_run_dir = Path(args.run_dir).resolve()
    project_root = Path(__file__).resolve().parent
    cand1 = supplied_run_dir / "saved_models"
    cand2 = project_root / "saved_models"

    if cand1.exists():
        saved_models_dir = cand1
    elif cand2.exists():
        saved_models_dir = cand2
    else:
        raise FileNotFoundError(f"saved_models/ directory not found at {cand1} or {cand2}. Please point --run_dir correctly or create saved_models/.")

    best_dir = saved_models_dir / "best"
    epochs_dir = saved_models_dir / "epochs"

    if args.model_type == 'best':
        ckpt_path = best_dir / "best_model.pt"
    else:
        if args.epoch is None:
            raise ValueError("If model_type=='epoch' you must provide --epoch N")
        ckpt_path = epochs_dir / f"checkpoint_epoch{args.epoch}.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # point to the same YAML used in training; adjust if you used a different config
    config_yaml = "configs/kmflow_re1000_rs256.yml"
    cfg = load_config(config_yaml, device)

    if args.lhc != -1:
        cfg.model.in_channels = 1
        cfg.model.out_ch = 1
        cfg.data.image_size = 256

    # build model and diffusion
    model = Model(cfg).to(device)
    diffusion = Diffusion(cfg)

    # Load checkpoint with safe fallback (weights_only=False if needed)
    try:
        ckpt = torch.load(str(ckpt_path), map_location=device)
    except Exception:
        print("torch.load default failed; retrying with weights_only=False (trusted checkpoint).")
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        # assume the checkpoint is a bare state_dict
        model.load_state_dict(ckpt)

    model.eval()
    torch.manual_seed(args.seed)

    # sample noise and run ddpm
    noise = torch.randn((1, cfg.model.in_channels, cfg.data.image_size, cfg.data.image_size), device=device)
    with torch.no_grad():
        out = diffusion.ddpm(noise, model, args.steps, plot_prog=False)  # returns (1,C,H,W)

    # prepare output path (file or directory)
    out_path = prepare_output_path(args.out, args.model_type)

    # save image only (PNG). No .npy
    img = tensor_to_image(out[0])
    img.save(str(out_path), format='PNG')
    print(f"Saved reconstruction image to {out_path}")
