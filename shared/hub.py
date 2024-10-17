from huggingface_hub import hf_hub_download, HfApi
from safetensors.torch import load_model, save_model
import json
import os
import shutil


def push_to_hub(obj, repo_id, name, tmp_dir="tmp"):
    os.makedirs(tmp_dir, exist_ok=True)
    json.dump(vars(obj.config), open(f'{tmp_dir}/config.json', 'w'), indent=2)
    save_model(obj, f'{tmp_dir}/model.safetensors')

    HfApi().upload_folder(folder_path=tmp_dir, path_in_repo=name, repo_id=repo_id)
    shutil.rmtree(tmp_dir)


def from_pretrained(cls, repo_id, name):
    config_path = hf_hub_download(repo_id=repo_id, filename=f"{name}/config.json")
    model_path = hf_hub_download(repo_id=repo_id, filename=f"{name}/model.safetensors")

    model = cls.from_config(**json.load(open(config_path)))
    load_model(model, model_path)
    return model