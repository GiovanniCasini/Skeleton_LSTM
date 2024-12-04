import os
import logging
import hydra
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import DictConfig
import numpy as np

from renderer.matplotlib import MatplotlibRender
from renderer.humor import HumorRenderer


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYOPENGL_PLATFORM"] = "egl"

def T(x):
    import torch
    import numpy as np

    if isinstance(x, torch.Tensor):
        return x.permute(*torch.arange(x.ndim - 1, -1, -1))
    else:
        return x.transpose(*np.arange(x.ndim - 1, -1, -1))


def render(motion_path, out_path, y_is_z_axis=False, fps=20, video=False):
    motions = np.load(motion_path)
    
    if motions.shape[1] == 6890:
        renderer = HumorRenderer(fps=20.0, imw=720, imh=720, video=video)
    else:
        renderer = MatplotlibRender(canonicalize=True,colors=['black', 'magenta', 'red', 'green', 'blue'], figsize=4,fps=20.0,jointstype='guoh3djoints')

    print(f"The video will be renderer there: {out_path}")

    if len(motions) == 1:
        motions = motions[0]

    if y_is_z_axis:
        x, mz, my = T(motions)
        motions = T(np.stack((x, -my, mz), axis=0))

    renderer(motions, title="", output=out_path, fps=fps, video=video)


if __name__ == "__main__":
    ids = ["03879"]
    video = True

    for id in ids:
        motion_path= f"{os.getcwd()}/outputs/SkeletonFormer_LossRec_KitML_m1_bs1_h1024_textEmbCLIP_DataSmpl__4l/Vertices/{id}.npy"
        
        sample_id = motion_path.split("/")[-1].replace(".npy", "")
        model_name = [a for a in motion_path.split("/") if "Skeleton" in a and "Loss" in a][0]
        out_path = f"{os.getcwd()}/visualizations/{model_name}/{sample_id}.mp4"

        Path(f"{os.getcwd()}/visualizations/{model_name}").mkdir(parents=True, exist_ok=True)
        
        render(motion_path, out_path, video=video)
