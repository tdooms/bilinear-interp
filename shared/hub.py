from huggingface_hub import hf_hub_download, HfApi
from safetensors.torch import load_model, save_model
import json
import os
import shutil

class HubMixin:
    """HF Hub Helper class. Assumes the model to have a `config` attribute and a `from_config` method."""
    def __init__(self) -> None:
        pass
    
    def push_to_hub(self, repo_id, name, tmp_dir="tmp"):
        os.makedirs(tmp_dir, exist_ok=True)
        json.dump(vars(self.config), open(f'{tmp_dir}/config.json', 'w'), indent=2)
        save_model(self, f'{tmp_dir}/model.safetensors')

        HfApi().upload_folder(folder_path=tmp_dir, path_in_repo=name, repo_id=repo_id)
        shutil.rmtree(tmp_dir)
    
    @staticmethod
    def from_pretrained(cls, repo_id, name):
        config_path = hf_hub_download(repo_id=repo_id, filename=f"{name}/config.json")
        model_path = hf_hub_download(repo_id=repo_id, filename=f"{name}/model.safetensors")

        sae = cls.from_config(**json.load(open(config_path)))
        load_model(sae, model_path)
        return sae