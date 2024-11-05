import src.prepare  # noqa

#from .stmc import read_timelines

from .metrics import calculate_activation_statistics_normalized
from .load_tmr_model import load_tmr_model_easy
# from .load_mtt_texts_motions import load_mtt_texts_motions
from .stmc_metrics import get_gt_metrics # , get_exp_metrics, print_latex

import os
import numpy as np 
import json
import orjson
from src.guofeats import joints_to_guofeats
from src.guofeats.motion_representation_local import (
    guofeats_to_joints as guofeats_to_joints_local,
)
import torch
from .metrics import (
    calculate_frechet_distance,
    calculate_activation_statistics_normalized,
    calculate_diversity,
)
from src.model.tmr import get_sim_matrix

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation


def numpy_to_video(np_data, save_path):
    # np_data: numpy array of predictions body movements
    # save_path: where the mp4 video will be saved
    num_frames = np_data.shape[0]
    num_joints = np_data.shape[1]

    body_connections = [(0,3), (3,6), (6,9), (12, 15), (23,21), (14, 17), (17, 19), (21,19), (13, 16), (16,18), (18,20) ,(0,2), (2,5), (5,8), (8,11), (0,1), (1,4), (4,7), (7,10)]

    # Crea una figura e un asse 3D
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # Set camera height
    ax.view_init(elev=10.)

    min_x, max_x = min(np_data[:,:,0].flatten()), max(np_data[:,:,0].flatten())
    min_y, max_y = min(np_data[:,:,1].flatten()), max(np_data[:,:,1].flatten())
    min_z, max_z = min(np_data[:,:,2].flatten()), max(np_data[:,:,2].flatten())

    # Inizializzazione della funzione di trama 
    def init():
        pass

    # Funzione di animazione, qui devi mettere il codice per aggiornare il tuo tracciato
    def update(num_frame):
        ax.cla()  # pulisce l'attuale ax
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_zlim(min_z, max_z)
        # print(f"Frame {num_frame} - x[0]: {data[num_frame,0,0]}")
        ax.scatter(np_data[num_frame,:,0], np_data[num_frame,:,1], np_data[num_frame,:,2], marker="x")

        for c1, c2 in body_connections:
            plt.plot([np_data[num_frame,c1,0], np_data[num_frame,c2,0]], [np_data[num_frame,c1,1], np_data[num_frame,c2,1]], [np_data[num_frame,c1,2], np_data[num_frame,c2,2]])
        
        for i in range(num_joints):
            ax.text(np_data[num_frame,i,0], np_data[num_frame,i,1], np_data[num_frame,i,2], str(i))

    # Crea l'animazione utilizzando le funzioni di inizializzazione e aggiornamento
    ani = animation.FuncAnimation(fig, update, frames=range(num_frames), init_func=init, blit=False)

    ax.set_xlabel("x axis")
    ax.set_ylabel("y axis")
    ax.set_zlabel("z axis")

    plt.show()

    # Salva l'animazione come file mp4, bisogna avere ffmepg installato
    ani.save(save_path, writer='ffmpeg')


def plot_points(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x,y,zs = z, s=20)
    count = 0
    for xi, yi, zi in zip(x, y, z):
        text = str(count)
        ax.text(xi, yi, zi, text, zdir=(1, 1, 1))
        count += 1

    ax.set_xlabel("x axis")
    ax.set_ylabel("y axis")
    ax.set_zlabel("z axis")

    plt.savefig("/andromeda/personal/lmandelli/stmc/plot.png", dpi=150)


def print_result(result):

    pp = ""
    for r in result:
        if type(r) is str:
            word = str(r)
            spaces=  (15 - len(word)) * " " 
        else:
            word = str(round(r, 3))
            spaces =  (15 - len(word)) * " "
        pp = pp + word + spaces + "- "

    print(pp)


def T(x):
    if isinstance(x, torch.Tensor):
        return x.permute(*torch.arange(x.ndim - 1, -1, -1))
    else:
        return x.transpose(*np.arange(x.ndim - 1, -1, -1))


def get_metrics(
    tmr_forward,
    text_dico,
    gt_mu,
    gt_cov,
    text_latents_gt,
    motion_latents_gt,
    generations_folder,
    infos, 
    diversity_times
):
    metrics = {}
    # Motion-to-text retrieval metrics
    m2t_top_1_lst = []
    m2t_top_3_lst = []

    # TMR scores
    m2m_score_lst = []
    m2t_score_lst = []

    # Transition distance
    trans_dist_lst = []

    # Store motion latents for FID+
    fid_realism_crop_motion_latents_lst = []

    for key, path_motion in infos.items():
        
        texts = [path_motion["text"]]
        # path = os.path.join(amass_folder, path_motion["motion_path"] + ".npy") # ground truth 
        path = os.path.join(generations_folder, key + ".npy") 

        motion = np.load(path)
        x, y, z = motion.T
        joint = np.stack((x, z, -y), axis=0).T
        motion_guofeats = joints_to_guofeats(joint)

        # gt_motion_guofeats = path_motion["motion"]
        joints_local = guofeats_to_joints_local(torch.from_numpy(motion_guofeats))
        joints_local = joints_local - joints_local[:, [0]]
        
        N = len(joints_local)
        inter_points = np.array([N // 4, 2 * N // 4, 3 * N // 4])

        trans_dist_lst.append(
            torch.linalg.norm(
                (joints_local[inter_points] - joints_local[inter_points - 1]),
                dim=-1,
            )
            .mean(-1)
            .flatten()
        )

        ### REALISM FID+
        nb_samples = 1
        realism_crop_motions = [motion_guofeats] # [motion_guofeats[x : x + n_real_nframes] for x in realism_idx]
        realism_crop_motion_latents = tmr_forward(realism_crop_motions)
        fid_realism_crop_motion_latents_lst.append(realism_crop_motion_latents)

        ### SEMANTICS
        
        # do not use real crops but the entire sequence (less than 10s)        
        crop_latents = tmr_forward([motion_guofeats])
        sim_matrix_m2t = get_sim_matrix(crop_latents, text_latents_gt).numpy()
        sim_matrix_m2m = get_sim_matrix(crop_latents, motion_latents_gt).numpy()

        for idx_text, text in enumerate(texts):
            text_number = text_dico[text]

            m2t_score_lst.append((sim_matrix_m2t[idx_text, text_number] + 1) / 2)
            m2m_score_lst.append((sim_matrix_m2m[idx_text, text_number] + 1) / 2)

            asort_m2t = np.argsort(sim_matrix_m2t[idx_text])[::-1]
            m2t_top_1_lst.append(1 * (text_number in asort_m2t[:1]))
            m2t_top_3_lst.append(1 * (text_number in asort_m2t[:3]))

    motion_latents = torch.concatenate(fid_realism_crop_motion_latents_lst)
    mu, cov = calculate_activation_statistics_normalized(motion_latents.numpy())

    # FID+ metrics
    metrics["fid"] = calculate_frechet_distance(
        gt_mu.astype(float),
        gt_cov.astype(float),
        mu.astype(float),
        cov.astype(float),
    )

    # Motion-to-text retrieval metrics
    metrics["m2t_top_1"] = np.mean(m2t_top_1_lst)
    metrics["m2t_top_3"] = np.mean(m2t_top_3_lst)

    # TMR scores
    metrics["m2t_score"] = np.mean(m2t_score_lst)
    metrics["m2m_score"] = np.mean(m2m_score_lst)

    # Transition distance
    metrics["transition"] = torch.concatenate(trans_dist_lst).mean().cpu().numpy()  

    # metrics["diversity"] = calculate_diversity(motion_latents, diversity_times) # 300

    return metrics


def load_test_texts_motions(path_ids, path_all_texts, path_annotations, DEBUG=0):
    test_texts, motions = [], []
    id_to_text_path = {}

    # Load test ids
    with open(path_ids) as f:
        lines = f.readlines()
    
    if DEBUG != 0:
        lines = lines[:DEBUG] 
    
    for l in lines:
        id = l.strip() 
        id_to_text_path[id] = {
            "text": None,
            "motion_path": None,
            "motion": None
        }
    # Load test texts
    with open(path_all_texts) as f:
        lines_t = f.readlines()
    for l in lines_t:
        idx = l.split("-")[0].strip()
        text = l.split("-")[1].strip()
        if idx in id_to_text_path.keys(): # il  test set non ha tutti i testi
            assert id_to_text_path[idx]["text"] is None
            id_to_text_path[idx]["text"] = text    
    # Load motion paths
    with open(path_annotations, "rb") as ff:
        annotations = orjson.loads(ff.read())
    for idx in id_to_text_path.keys():
        path = annotations[idx]["path"]
        id_to_text_path[idx]["motion_path"] = path
    # remove humanact12
    id_to_text_path = {k: v for k, v in id_to_text_path.items() if "humanact12" not in v["motion_path"]}# Giusto o sbagliato?
    # Load motions
    for idx in id_to_text_path.keys():
        amass_path = id_to_text_path[idx]["motion_path"]
        path = os.path.join(amass_folder, amass_path + ".npy")
        motion = np.load(path)
        # Croppo i movimenti maggiori di 10 secondi prendendo gli estremi specificati nel file annotations
        start_frame = int(annotations[idx]["annotations"][0]["start"] * fps)
        end_frame = int(annotations[idx]["annotations"][0]["end"] * fps)
        motion = motion[start_frame:end_frame]

        x, y, z = motion.T
        joint = np.stack((x, z, -y), axis=0).T
        feats = joints_to_guofeats(joint)
        id_to_text_path[idx]["motion"] = feats

    test_texts = [id_to_text_path[i]["text"] for i in id_to_text_path.keys()]
    motions = [id_to_text_path[i]["motion"] for i in id_to_text_path.keys()]
    
    return test_texts, motions, id_to_text_path


### BASIC PATHS

amass_folder = "../datasets/motions/AMASS_20.0_fps_nh_smpljoints_neutral_nobetas"
base_path_splitme_humanml3d = "/andromeda/personal/lmandelli/stmc/pretrained_models/mdm-smpl_splitme_humanml3d"
base_path_splitme_kitml = "/andromeda/personal/lmandelli/stmc/pretrained_models/mdm-smpl_splitme_kitml"
base_path_humanml_clip = "/andromeda/personal/lmandelli/stmc/pretrained_models/mdm-smpl_clip_smplrifke_humanml3d"
base_path_kitml_clip = "/andromeda/personal/lmandelli/stmc/pretrained_models/mdm-smpl_clip_smplrifke_kitml"

exp_gt = {
        "name": "gt",
        "generations_folder": f"{amass_folder}/"
}
exp_text_humanml = {
        "name": "text",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/splits/me/test.txt",
        "generations_folder": f"{base_path_splitme_humanml3d}/generations_text/", 
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/splits/me/annotations_test.json"
}
exp_timeline_humanml = {
        "name": "timeline",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/splits/me/test.txt",
        "generations_folder": f"{base_path_splitme_humanml3d}/generations_timeline/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/splits/me/annotations_test.json"
}
exp_stmc_humanml ={
        "name": "stmc",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/splits/me/test.txt",
        "generations_folder":f"{base_path_splitme_humanml3d}/generations_timeline_stmc/", 
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/splits/me/annotations_test.json"
}
exp_timeline_humanml_w1 = {
        "name": "timeline_w1",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/splits/me/test.txt",
        "generations_folder": f"{base_path_splitme_humanml3d}/generations_timeline_w1/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/splits/me/annotations_test.json"
}
exp_timeline_humanml_w2 = {
        "name": "timeline_w2",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/splits/me/test.txt",
        "generations_folder": f"{base_path_splitme_humanml3d}/generations_timeline_w2/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/splits/me/annotations_test.json"
}
exp_timeline_humanml_w3 = {
        "name": "timeline_w3",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/splits/me/test.txt",
        "generations_folder": f"{base_path_splitme_humanml3d}/generations_timeline_w3/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/splits/me/annotations_test.json"
}
exp_timeline_humanml_w4 = {
        "name": "timeline_w4",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/splits/me/test.txt",
        "generations_folder": f"{base_path_splitme_humanml3d}/generations_timeline_w4/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/splits/me/annotations_test.json"
}
exp_timeline_humanml_w5 = {
        "name": "timeline_w5",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/splits/me/test.txt",
        "generations_folder": f"{base_path_splitme_humanml3d}/generations_timeline_w5/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/splits/me/annotations_test.json"
}
exp_timeline_humanml_w6 = {
        "name": "timeline_w6",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/splits/me/test.txt",
        "generations_folder": f"{base_path_splitme_humanml3d}/generations_timeline_w6/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/splits/me/annotations_test.json"
}
exp_timeline_humanml_w7 = {
        "name": "timeline_w7",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/splits/me/test.txt",
        "generations_folder": f"{base_path_splitme_humanml3d}/generations_timeline_w7/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/splits/me/annotations_test.json"
}
exp_timeline_humanml_w8 = {
        "name": "timeline_w8",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/splits/me/test.txt",
        "generations_folder": f"{base_path_splitme_humanml3d}/generations_timeline_w8/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/splits/me/annotations_test.json"
}
exp_timeline_humanml_w9 = {
        "name": "timeline_w9",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/splits/me/test.txt",
        "generations_folder": f"{base_path_splitme_humanml3d}/generations_timeline_w9/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/splits/me/annotations_test.json"
}
exp_timeline_humanml_w10 = {
        "name": "timeline_w10",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/splits/me/test.txt",
        "generations_folder": f"{base_path_splitme_humanml3d}/generations_timeline_w10/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/splits/me/annotations_test.json"
}
exp_timeline_humanml_w15 = {
        "name": "timeline_w15",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/splits/me/test.txt",
        "generations_folder": f"{base_path_splitme_humanml3d}/generations_timeline_w15/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/splits/me/annotations_test.json"
}
exp_multitext_text_humanml ={
        "name":"multi_text",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/splits/test.txt", 
        "generations_folder":  f"{base_path_humanml_clip}/generations_multitext_text/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/annotations.json"
}
exp_multitext_timeline_humanml = {
        "name": "multi_t",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/splits/test.txt", 
        "generations_folder": f"{base_path_humanml_clip}/generations_multitext_timeline/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/annotations.json"
}
exp_multitext_timeline_humanml_w1 = {
        "name": "multi_t_w1",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/splits/test.txt", 
        "generations_folder": f"{base_path_humanml_clip}/generations_multitext_timeline_w1/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/annotations.json"
}
exp_multitext_timeline_humanml_w2 = {
        "name": "multi_t_w2",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/splits/test.txt", 
        "generations_folder": f"{base_path_humanml_clip}/generations_multitext_timeline_w2/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/annotations.json"
}
exp_multitext_timeline_humanml_w3 = {
        "name": "multi_t_w3",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/splits/test.txt", 
        "generations_folder": f"{base_path_humanml_clip}/generations_multitext_timeline_w3/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/annotations.json"
}
exp_multitext_timeline_humanml_w4 = {
        "name": "multi_t_w4",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/splits/test.txt", 
        "generations_folder": f"{base_path_humanml_clip}/generations_multitext_timeline_w4/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/annotations.json"
}
exp_multitext_timeline_humanml_w5 = {
        "name": "multi_t_w5",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/splits/test.txt", 
        "generations_folder": f"{base_path_humanml_clip}/generations_multitext_timeline_w5/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/annotations.json"
}
exp_multitext_timeline_humanml_w6 = {
        "name": "multi_t_w6",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/splits/test.txt", 
        "generations_folder": f"{base_path_humanml_clip}/generations_multitext_timeline_w6/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/annotations.json"
}
exp_multitext_timeline_humanml_w7 = {
        "name": "multi_t_w7",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/splits/test.txt", 
        "generations_folder": f"{base_path_humanml_clip}/generations_multitext_timeline_w7/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/annotations.json"
}
exp_multitext_timeline_humanml_w8 = {
        "name": "multi_t_w8",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/splits/test.txt", 
        "generations_folder": f"{base_path_humanml_clip}/generations_multitext_timeline_w8/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/annotations.json"
}
exp_multitext_timeline_humanml_w9 = {
        "name": "multi_t_w9",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/splits/test.txt", 
        "generations_folder": f"{base_path_humanml_clip}/generations_multitext_timeline_w9/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/annotations.json"
}
exp_multitext_timeline_humanml_w10 = {
        "name": "multi_t_w10",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/splits/test.txt", 
        "generations_folder": f"{base_path_humanml_clip}/generations_multitext_timeline_w10/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/annotations.json"
}
exp_multitext_timeline_humanml_w15 = {
        "name": "multi_t_w15",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/splits/test.txt", 
        "generations_folder": f"{base_path_humanml_clip}/generations_multitext_timeline_w15/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/annotations.json"
}
exp_multitext_timeline_humanml_w20 = {
        "name": "multi_t_w20",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/splits/test.txt", 
        "generations_folder": f"{base_path_humanml_clip}/generations_multitext_timeline_w20/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/annotations.json"
}

exp_text_kitml = {
        "name": "text",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/me/test.txt",
        "generations_folder": f"{base_path_splitme_kitml}/generations_text/", 
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/me/annotations_test.json"
}
exp_timeline_kitml = {
        "name": "timeline",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/me/test.txt",
        "generations_folder": f"{base_path_splitme_kitml}/generations_timeline/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/me/annotations_test.json"
}
exp_stmc_kitml ={
        "name": "timeline_stmc",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/me/test.txt",
        "generations_folder":f"{base_path_splitme_kitml}/generations_timeline_stmc/", 
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/me/annotations_test.json"
}
exp_timeline_kitml_w1 = {
        "name": "timeline w1",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/me/test.txt",
        "generations_folder": f"{base_path_splitme_kitml}/generations_timeline_w1/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/me/annotations_test.json"
}
exp_timeline_kitml_w2 = {
        "name": "timeline w2",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/me/test.txt",
        "generations_folder": f"{base_path_splitme_kitml}/generations_timeline_w2/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/me/annotations_test.json"
}
exp_timeline_kitml_w3 = {
        "name": "timeline w3",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/me/test.txt",
        "generations_folder": f"{base_path_splitme_kitml}/generations_timeline_w3/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/me/annotations_test.json"
}
exp_timeline_kitml_w4 = {
        "name": "timeline w4",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/me/test.txt",
        "generations_folder": f"{base_path_splitme_kitml}/generations_timeline_w4/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/me/annotations_test.json"
}
exp_timeline_kitml_w5 = {
        "name": "timeline w5",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/me/test.txt",
        "generations_folder": f"{base_path_splitme_kitml}/generations_timeline_w5/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/me/annotations_test.json"
}
exp_timeline_kitml_w6 = {
        "name": "timeline w6",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/me/test.txt",
        "generations_folder": f"{base_path_splitme_kitml}/generations_timeline_w6/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/me/annotations_test.json"
}
exp_timeline_kitml_w7 = {
        "name": "timeline w7",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/me/test.txt",
        "generations_folder": f"{base_path_splitme_kitml}/generations_timeline_w7/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/me/annotations_test.json"
}
exp_timeline_kitml_w8 = {
        "name": "timeline w8",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/me/test.txt",
        "generations_folder": f"{base_path_splitme_kitml}/generations_timeline_w8/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/me/annotations_test.json"
}
exp_timeline_kitml_w9 = {
        "name": "timeline w9",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/me/test.txt",
        "generations_folder": f"{base_path_splitme_kitml}/generations_timeline_w9/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/me/annotations_test.json"
}
exp_timeline_kitml_w10 = {
        "name": "timeline w10",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/me/test.txt",
        "generations_folder": f"{base_path_splitme_kitml}/generations_timeline_w10/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/me/annotations_test.json"
}
exp_timeline_kitml_w12 = {
        "name": "timeline w12",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/me/test.txt",
        "generations_folder": f"{base_path_splitme_kitml}/generations_timeline_w12/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/me/annotations_test.json"
}
exp_timeline_kitml_w14 = {
        "name": "timeline w14",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/me/test.txt",
        "generations_folder": f"{base_path_splitme_kitml}/generations_timeline_w14/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/me/annotations_test.json"
}
exp_timeline_kitml_w15 = {
        "name": "timeline w15",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/me/test.txt",
        "generations_folder": f"{base_path_splitme_kitml}/generations_timeline_w15/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/me/annotations_test.json"
}
exp_timeline_kitml_w16 = {
        "name": "timeline w16",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/me/test.txt",
        "generations_folder": f"{base_path_splitme_kitml}/generations_timeline_w16/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/me/annotations_test.json"
}
exp_timeline_kitml_w18 = {
        "name": "timeline w18",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/me/test.txt",
        "generations_folder": f"{base_path_splitme_kitml}/generations_timeline_w18/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/me/annotations_test.json"
}
exp_timeline_kitml_w19 = {
        "name": "timeline w19",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/me/test.txt",
        "generations_folder": f"{base_path_splitme_kitml}/generations_timeline_w19/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/me/annotations_test.json"
}
exp_timeline_kitml_w20 = {
        "name": "timeline w20",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/me/test.txt",
        "generations_folder": f"{base_path_splitme_kitml}/generations_timeline_w20/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/me/annotations_test.json"
}
exp_timeline_kitml_w22 = {
        "name": "timeline w22",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/me/test.txt",
        "generations_folder": f"{base_path_splitme_kitml}/generations_timeline_w22/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/me/annotations_test.json"
}
exp_timeline_kitml_w25 = {
        "name": "timeline w25",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/me/test.txt",
        "generations_folder": f"{base_path_splitme_kitml}/generations_timeline_w25/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/me/annotations_test.json"
}
exp_multitext_text_kitml ={
        "name":"multi_text_k",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/test.txt", 
        "generations_folder":  f"{base_path_kitml_clip}/generations_multitext_text/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/annotations.json"
}
exp_multitext_kitml_timeline = {
        "name": "multi_k_t",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/test.txt", 
        "generations_folder": f"{base_path_kitml_clip}/generations_multitext_timeline/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/annotations.json"
}
exp_multitext_kitml_timeline_w1 = {
        "name": "multi_k_t_w1",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/test.txt", 
        "generations_folder": f"{base_path_kitml_clip}/generations_multitext_timeline_w1/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/annotations.json"
}
exp_multitext_kitml_timeline_w2 = {
        "name": "multi_k_t_w2",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/test.txt", 
        "generations_folder": f"{base_path_kitml_clip}/generations_multitext_timeline_w2/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/annotations.json"
}
exp_multitext_kitml_timeline_w3 = {
        "name": "multi_k_t_w3",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/test.txt", 
        "generations_folder": f"{base_path_kitml_clip}/generations_multitext_timeline_w3/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/annotations.json"
}
exp_multitext_kitml_timeline_w4 = {
        "name": "multi_k_t_w4",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/test.txt", 
        "generations_folder": f"{base_path_kitml_clip}/generations_multitext_timeline_w4/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/annotations.json"
}
exp_multitext_kitml_timeline_w5 = {
        "name": "multi_k_t_w5",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/test.txt", 
        "generations_folder": f"{base_path_kitml_clip}/generations_multitext_timeline_w5/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/annotations.json"
}
exp_multitext_kitml_timeline_w6 = {
        "name": "multi_k_t_w6",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/test.txt", 
        "generations_folder": f"{base_path_kitml_clip}/generations_multitext_timeline_w6/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/annotations.json"
}
exp_multitext_kitml_timeline_w7 = {
        "name": "multi_k_t_w7",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/test.txt", 
        "generations_folder": f"{base_path_kitml_clip}/generations_multitext_timeline_w7/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/annotations.json"
}
exp_multitext_kitml_timeline_w8 = {
        "name": "multi_k_t_w8",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/test.txt", 
        "generations_folder": f"{base_path_kitml_clip}/generations_multitext_timeline_w8/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/annotations.json"
}
exp_multitext_kitml_timeline_w9 = {
        "name": "multi_k_t_w9",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/test.txt", 
        "generations_folder": f"{base_path_kitml_clip}/generations_multitext_timeline_w9/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/annotations.json"
}
exp_multitext_kitml_timeline_w10 = {
        "name": "multi_k_t_w10",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/test.txt", 
        "generations_folder": f"{base_path_kitml_clip}/generations_multitext_timeline_w10/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/annotations.json"
}
exp_multitext_kitml_timeline_w15 = {
        "name": "multi_k_t_w15",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/test.txt", 
        "generations_folder": f"{base_path_kitml_clip}/generations_multitext_timeline_w15/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/annotations.json"
}
exp_multitext_kitml_timeline_w20 = {
        "name": "multi_k_t_w20",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/splits/test.txt", 
        "generations_folder": f"{base_path_kitml_clip}/generations_multitext_timeline_w20/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/kitml/annotations.json"
}
exp_timeline_humanml_new = {
        "name": "timeline_new",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/splits/me/test.txt",
        "generations_folder": f"{base_path_splitme_humanml3d}/generations_timeline_new/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/splits/me/annotations_test.json"
}
exp_timeline_humanml_linear_weight_scheduler = {
        "name": "linear w sched",
        "path_ids":f"/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/splits/me/test.txt",
        "generations_folder": f"{base_path_splitme_humanml3d}/generations_timeline_linear/",
        "path_annotations": "/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/splits/me/annotations_test.json"
}

### SETTINGS 
input_types = [exp_gt, exp_text_kitml, exp_timeline_kitml_w7]

input_types = [exp_gt, exp_stmc_kitml, exp_text_kitml, exp_timeline_kitml]
input_types = [exp_gt, exp_multitext_text_kitml, exp_multitext_kitml_timeline]
input_types = [exp_gt, exp_text_humanml, exp_stmc_humanml, exp_timeline_humanml, exp_timeline_humanml_w7] 
input_types = [exp_gt, exp_multitext_text_humanml, exp_multitext_timeline_humanml_w1, exp_multitext_timeline_humanml_w2, exp_multitext_timeline_humanml_w3, exp_multitext_timeline_humanml_w4, exp_multitext_timeline_humanml_w5, exp_multitext_timeline_humanml_w6, exp_multitext_timeline_humanml_w7, exp_multitext_timeline_humanml_w8, exp_multitext_timeline_humanml_w9, exp_multitext_timeline_humanml_w10]
input_types = [exp_gt, exp_multitext_text_kitml, exp_multitext_kitml_timeline, exp_multitext_kitml_timeline_w6]
input_types = [exp_gt, exp_multitext_text_humanml, exp_multitext_timeline_humanml] 
input_types = [exp_gt, exp_timeline_humanml, exp_timeline_humanml_new] #, exp_stmc_humanml, exp_timeline_humanml] 
input_types = [exp_gt, exp_text_kitml, exp_stmc_kitml, exp_timeline_kitml, exp_timeline_kitml_w8]


# tabella 3 - KITML
input_types = [exp_gt, exp_timeline_kitml_w1, exp_timeline_kitml_w5, exp_timeline_kitml_w10, exp_timeline_kitml_w15, exp_timeline_kitml_w20]
# GT-vs-Text-vs-STMC-vs-Oursv - KITML
input_types = [exp_gt, exp_text_kitml, exp_stmc_kitml, exp_timeline_kitml_w18] 

# Ablation splitme on kitml 
input_types = [exp_gt, exp_text_kitml, exp_stmc_kitml, exp_timeline_kitml_w1, exp_timeline_kitml_w2, exp_timeline_kitml_w3, exp_timeline_kitml_w4, exp_timeline_kitml_w5, exp_timeline_kitml_w6, exp_timeline_kitml_w7, exp_timeline_kitml_w8, exp_timeline_kitml_w9, exp_timeline_kitml_w10, exp_timeline_kitml_w12, exp_timeline_kitml_w14, exp_timeline_kitml_w15, exp_timeline_kitml_w16, exp_timeline_kitml_w18, exp_timeline_kitml_w19, exp_timeline_kitml_w20, exp_timeline_kitml_w22, exp_timeline_kitml_w25] 

# Ablation multi-annotations on kitml
input_types = [exp_gt, exp_multitext_text_kitml, exp_multitext_kitml_timeline_w1, exp_multitext_kitml_timeline_w2, exp_multitext_kitml_timeline_w3, exp_multitext_kitml_timeline_w4, exp_multitext_kitml_timeline_w5, exp_multitext_kitml_timeline_w6, exp_multitext_kitml_timeline_w7, exp_multitext_kitml_timeline_w8, exp_multitext_kitml_timeline_w9, exp_multitext_kitml_timeline_w10, exp_multitext_kitml_timeline_w15, exp_multitext_kitml_timeline_w20]

# Ablation splitme on humanml
input_types = [exp_gt, exp_stmc_humanml, exp_timeline_humanml_w15, exp_text_humanml, exp_timeline_humanml_w1, exp_timeline_humanml_w2, exp_timeline_humanml_w3, exp_timeline_humanml_w4, exp_timeline_humanml_w5, exp_timeline_humanml_w6, exp_timeline_humanml_w7, exp_timeline_humanml_w8, exp_timeline_humanml_w9, exp_timeline_humanml_w10]

# Ablation multi-annotations on humanml
input_types = [exp_gt, exp_multitext_timeline_humanml_w1, exp_multitext_timeline_humanml_w2, exp_multitext_timeline_humanml_w3, exp_multitext_timeline_humanml_w4, exp_multitext_timeline_humanml_w5, exp_multitext_timeline_humanml_w6, exp_multitext_timeline_humanml_w7, exp_multitext_timeline_humanml_w8, exp_multitext_timeline_humanml_w9, exp_multitext_timeline_humanml_w10]

# gne - humanml
input_types = [exp_gt, exp_timeline_humanml_w5, exp_text_humanml, exp_stmc_humanml]
# gne - kitml
input_types = [exp_gt, exp_timeline_kitml_w10, exp_text_kitml, exp_stmc_kitml]

input_types = [exp_timeline_humanml, exp_timeline_humanml_linear_weight_scheduler]

DEBUG = 0
save_scores = False

###
np.random.seed(0)
device = "cpu"
fps = 20.0
path_ids = input_types[1]["path_ids"]
assert all([inp["path_ids"]==path_ids for inp in input_types[1:]])
path_annotations = input_types[1]["path_annotations"]
assert all([inp["path_annotations"]==path_annotations for inp in input_types[1:]])
dataset = "humanml3d" if "humanml3d" in input_types[1]["path_ids"] else "kitml"

result = []

names = [i["name"] for i in input_types]
print(f"\nExperiment on {names} and with debug [{DEBUG}] of the split [{path_ids}]\n")

with open(path_ids) as f:
    ids_gt = f.readlines()
ids_gt = [i.strip() for i in ids_gt]

annotations = json.load(open(path_annotations))
ids_gt = [i for i in ids_gt if  "humanact12" not in annotations[i]["path"]]
# filter annotations
# ids_gt = [i for i in ids_gt if (annotations[i]["annotations"][0]["end"] - annotations[i]["annotations"][0]["start"] <= 10) and (annotations[i]["annotations"][0]["end"] - annotations[i]["annotations"][0]["start"] >= 2)]
annotations = {k: v for k, v in annotations.items() if "humanact12" not in v["path"]} # TODO Giusto o sbagliato? (metterlo o no in KITML non cambia nulla)

if DEBUG != 0:
    ids_gt = ids_gt[:DEBUG]

texts_gt, motions_guofeats_gt = [], []
for idx in ids_gt:
    value = annotations[idx]
    # prendo il testo della prima annotazione testuale
    texts_gt.append(value["annotations"][-1]["text"])

    motion_path = exp_gt["generations_folder"] + value["path"]+".npy"
    motion = np.load(motion_path) # [num_frames, 24, 3], con 0 < num_frames   
    start_frame = int(value["annotations"][0]["start"] * fps)
    end_frame = int(value["annotations"][0]["end"] * fps)
    # end_frame = end_frame if end_frame - start_frame <= 200 else start_frame+200 # TODO TOGLIERE | è unfair? 
    motion = motion[start_frame:end_frame] # [num_frames, 24, 3], con 0 < num_frames <= 200, ovvero 10 sec * 20 fps  
    # assert motion.shape[0] <= 10*fps

    x, y, z = motion.T
    joint = np.stack((x, z, -y), axis=0).T # [num_frames, 24, 3]
    feats = joints_to_guofeats(joint) # [num_frames, 263]
    motions_guofeats_gt.append(feats)

tmr_forward = load_tmr_model_easy(device, dataset)

print(f"texts_gt len {len(texts_gt)}")
diversity_times = 300 if 300 < len(texts_gt) else len(texts_gt) - 1

# texts_gt: list(str) - N elementi per testset
# motions_guofeats_gt: list: N * tensor(n_frames, 263)) - N elementi per testset
text_latents_gt = tmr_forward(texts_gt) #  tensor(N, 256)
motion_latents_gt = tmr_forward(motions_guofeats_gt)  # tensor(N, 256)
print(f"text_latents_gt shape {text_latents_gt.shape}")
print(f"motion_latents_gt shape {motion_latents_gt.shape}")
gt_mu, gt_cov = calculate_activation_statistics_normalized(motion_latents_gt.numpy())

result.append(["Experiment", "R1", "R3", "R10", "M2T", "M2M", "FID", "Trans"])
print_result(result[0])

for experiment in input_types:

    dic_m2t = {}
    dic_m2m = {}

    # Load the motions
    motions_guofeats = []
    for idx in ids_gt:

        if experiment["name"] == "gt":
            motion_path = experiment["generations_folder"] + annotations[idx]["path"]+".npy"
            motion = np.load(motion_path) # [num_frames, 24, 3], con 0 < num_frames   
            start_frame = int(annotations[idx]["annotations"][0]["start"] * fps)
            end_frame = int(annotations[idx]["annotations"][0]["end"] * fps)
            # end_frame = end_frame if end_frame - start_frame <= 200 else start_frame+200# TOGLIERE
            motion = motion[start_frame:end_frame] # [num_frames, 24, 3], con 0 < num_frames <= 200, ovvero 10 sec * 20 fps  
            # assert motion.shape[0] <= 10*fps
        else:
            motion_path = experiment["generations_folder"] + idx +".npy"
            motion = np.load(motion_path)
            if motion.shape[0] > 200:
                motion = motion[:200]

        x, y, z = motion.T
        joint = np.stack((x, z, -y), axis=0).T # [num_frames, 24, 3]
        feats = joints_to_guofeats(joint) # [num_frames, 263]
        motions_guofeats.append(feats)
    
    max_num_frames = max([m.shape[0] for m in motions_guofeats])
    # assert max_num_frames <= 10*fps
    # print(f"motions_guofeats len {len(motions_guofeats)}")

    motion_latents = tmr_forward(motions_guofeats)  # tensor(N, 256)
    sim_matrix_m2t = get_sim_matrix(motion_latents, text_latents_gt).numpy()
    sim_matrix_m2m = get_sim_matrix(motion_latents, motion_latents_gt).numpy()

    # motion-to-text retrieval metrics
    m2t_top_1_lst = []
    m2t_top_3_lst = []
    m2t_top_10_lst = []
    # TMR motion-to-motion (M2M) score
    m2t_score_lst = []
    m2m_score_lst = []
    
    for idx in range(len(sim_matrix_m2t)):

        # score between 0 and 1
        m2t_score_lst.append((sim_matrix_m2t[idx, idx] + 1) / 2)
        m2m_score_lst.append((sim_matrix_m2m[idx, idx] + 1) / 2)
        
        asort = np.argsort(sim_matrix_m2t[idx])[::-1]
        m2t_top_1_lst.append(1 * (idx in asort[:1]))
        m2t_top_3_lst.append(1 * (idx in asort[:3]))
        m2t_top_10_lst.append(1 * (idx in asort[:10]))

        dic_m2t[ids_gt[idx]] = (sim_matrix_m2t[idx, idx] + 1) / 2
        dic_m2m[ids_gt[idx]] = (sim_matrix_m2m[idx, idx] + 1) / 2

    m2t_top_1 = np.mean(m2t_top_1_lst)
    m2t_top_3 = np.mean(m2t_top_3_lst)
    m2t_top_10 = np.mean(m2t_top_10_lst)
    m2t_score = np.mean(m2t_score_lst)
    m2m_score = np.mean(m2m_score_lst)

    # Transition distance:
    trans_dist_lst = []
    for motion_guofeats in motions_guofeats:
        # for the text baseline for example
        N = len(motion_guofeats)
        inter_points = np.array([N // 4, 2 * N // 4, 3 * N // 4]) # prendo tre frame 

        gt_motion_guofeats = torch.from_numpy(motion_guofeats) # (n_frames, 263)
        gt_joints_local = guofeats_to_joints_local(gt_motion_guofeats)
        
        gt_joints_local = gt_joints_local - gt_joints_local[:, [0]] # (n_frames, 22, 3) # TODO questa mi sembra sbagliata: gt_joints_local[:, [0]] è una matrice [num_frame, 1, 3]...? 

        # Same distance as in TEACH
        trans_dist_lst.append(
            torch.linalg.norm(
                (gt_joints_local[inter_points] - gt_joints_local[inter_points - 1]), # Guardo la distanza tra i tre frame i tre precedenti 
                dim=-1,
            )
            .mean(-1)
            .flatten()
        )

    # Transition distance
    transition = torch.concatenate(trans_dist_lst).mean().numpy().item()
    # diversity = calculate_diversity(motion_latents, diversity_times) 

    mu, cov = calculate_activation_statistics_normalized(motion_latents.numpy())

    # FID+ metrics
    fid = calculate_frechet_distance(
        gt_mu.astype(float),
        gt_cov.astype(float),
        mu.astype(float),
        cov.astype(float),
    )

    result.append( [experiment["name"], m2t_top_1*100, m2t_top_3*100, m2t_top_10*100, m2t_score, m2m_score, fid, transition*100 ] )
    print_result(result[-1])

    if save_scores:
        path_score = f"/andromeda/personal/lmandelli/stmc/neverseen_dataset/scores/{dataset}/{experiment['name']}/"
        with open(path_score + "m2m.json", 'w', encoding='utf-8') as f:
            json.dump(dic_m2m, f, ensure_ascii=False, indent=4)
        with open(path_score + "m2t.json", 'w', encoding='utf-8') as f:
            json.dump(dic_m2t, f, ensure_ascii=False, indent=4)

exit()