import torch
import sys
import os
from pathlib import Path
from torchvision import transforms

import PIL
from PIL import Image

sys.path.append("geo_predictors")
from geo_predictors.omnidata.modules.midas.dpt_depth import DPTDepthModel

downsampling = 1
img_size = 512


def resolve_omnidata_ckpt() -> str:
    env_path = os.environ.get("DREAMSCENE_OMNIDATA_CKPT")
    if env_path and Path(env_path).exists():
        return env_path

    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        repo_root / "pre_checkpoints" / "omnidata_dpt_depth_v2.ckpt",
        repo_root.parent.parent.parent / "DreamScene360" / "pre_checkpoints" / "omnidata_dpt_depth_v2.ckpt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return str(candidates[0])


ckpt_path = resolve_omnidata_ckpt()
model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=1)
model.to(torch.device('cpu'))
checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
if 'state_dict' in checkpoint:
    state_dict = {}
    for k, v in checkpoint['state_dict'].items():
        state_dict[k[6:]] = v
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)
trans_totensor = transforms.Compose([transforms.Normalize(mean=0.5, std=0.5)])

def estimate_depth(img, mode='test'):
    h, w = img.shape[1:3]
    img = img.unsqueeze(0)
    model.to(torch.device('cuda'))
    img_tensor = trans_totensor(img)
    if mode == 'test':
        with torch.no_grad():
            prediction = model(img_tensor)
            prediction = prediction.squeeze()
    else:
        prediction = model(img_tensor).squeeze()
    return prediction
