import os
import sys
import yaml
import torch
import fnmatch
from tube_models import M2F_MiTB5_TumorBed, SegFormerBinary

root_code_dir = os.path.dirname(os.path.abspath(__file__))

def check_input_valid(input_path):
    if not os.path.exists(input_path):
        sys.exit(f"\nPath does not exist: {input_path}")
    if not os.path.isfile(input_path):
        sys.exit(f"\nExpected a file but got a directory: {input_path}")

    return True

def ensure_dir(d): 
    os.makedirs(d, exist_ok=True)

def load_app_config():
    """Load app_config.yaml relative to this script."""
    config_path = os.path.join(root_code_dir, "config", "app_config.yaml")
    with open(config_path, "r") as f:
        app_config = yaml.safe_load(f)
    return app_config


def build_model_and_path(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_dir = os.path.join(root_code_dir, cfg.model_dir, cfg.tube_ckpt)
    model_path = os.path.join(model_dir, 'model.pth')
    model_config_path = os.path.join(model_dir, "config.json")
    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    if cfg.tube_model_type.lower() == "mask2former":
        # instantiate model (same dims as training defaults)
        model = M2F_MiTB5_TumorBed(
            ckpt_name=model_config["backbone"],
            embed_dim=model_config["decoder_params"]["embed_dim"],
            num_queries=model_config["decoder_params"]["num_queries"],
            num_layers=model_config["decoder_params"]["num_layers"],
            nheads=model_config["decoder_params"]["nheads"]
        ).to(device)
    elif cfg.tube_model_type.lower() == "segformer":
        model = SegFormerBinary(ckpt=cfg.tube_ckpt, embed_dim=256).to(device)
    else:
        raise ValueError(f"Unknown model_type: {cfg.tube_model_type}")

    print(f"Loading [{cfg.tube_model_type}] weights from: {model_path}")
    state = torch.load(model_path, map_location=device)
    
    # allow either raw state_dict or checkpoint dict
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    
    return model, device, model_config

