#from transformers import AutoImageProcessor, Dinov2Model
import torch
import os
from pathlib import Path
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# from datasets import load_dataset
from torchvision.transforms import Compose
from torchvision import transforms
from dinov2.dinov2.hub.backbones import dinov2_vitb14 


def resolve_dino_weights() -> str:
    env_path = os.environ.get("DREAMSCENE_DINOV2_WEIGHTS")
    if env_path and Path(env_path).exists():
        return env_path

    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        repo_root / "checkpoints" / "dinov2_vitb14.pth",
        repo_root.parent.parent.parent / "DreamScene360" / "checkpoints" / "dinov2_vitb14.pth",
        Path("/public_f/peizhi/dinov2_vitb14/dinov2_vitb14.pth"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return str(candidates[0])


model = dinov2_vitb14(pretrained=False).cuda()
# model =  torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').cuda()

# model =  torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=False).cuda()

model.load_state_dict(torch.load(resolve_dino_weights(), map_location='cuda'))

def get_Feature_from_DinoV2(tensor, model = model):
    transform = Compose([
        transforms.Resize(504, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])



    trans_img = transform(tensor).unsqueeze(0)
    # feature = model.get_intermediate_layers(trans_img)
    feature = model(trans_img)

    # print(trans_img)
    # print(trans_img.shape)
    # print(feature)
    # print(feature[0].shape)
    return feature



