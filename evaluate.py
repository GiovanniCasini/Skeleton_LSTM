
from TMR.mtt.metrics import calculate_activation_statistics_normalized
from TMR.mtt.load_tmr_model import load_tmr_model_easy

import os
import numpy as np 
import json
import orjson
from TMR.src.guofeats import joints_to_guofeats
from TMR.src.guofeats.motion_representation_local import (
    guofeats_to_joints as guofeats_to_joints_local,
)
import torch
from TMR.mtt.metrics import (
    calculate_frechet_distance,
    calculate_activation_statistics_normalized,
)
from TMR.src.model.tmr import get_sim_matrix

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

from generation import load_input


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
    mu, cov = calculate_activation_statistics_normalized(motion_latents.cpu().numpy())

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


def main():    

    base_smpldataset_path = "/andromeda/personal/lmandelli/stmc/datasets"
    model_name = "SkeletonFormer_LossRec_HumML_m1_bs1_h256_textEmbCLIP_DataSmpl_4l_long"
    amass_folder = f"{base_smpldataset_path}/motions/AMASS_20.0_fps_nh_smpljoints_neutral_nobetas"
    data_format = "Smpl" if "Smpl" in model_name else "Joints"
    feature_size = 63 if data_format=="Joints" else 205
    dataset = "humanml3d" if "HumML" in model_name else "kitml"


    kit_joints_folder = f"{os.getcwd()}/kit_numpy/"
    generations_folder = f"{os.getcwd()}/outputs/{model_name}/Joints"

    path_annotations = f"{base_smpldataset_path}/annotations/{dataset}/annotations.json"
    annotations = json.load(open(path_annotations))

    exp_gt = {
            "name": "gt",
            "generations_folder": f"{kit_joints_folder}/" if (data_format=="Joints" and dataset=="kitml") else amass_folder
    }
    exp_text_kitml = {
            "name": "kitml",
            "generations_folder": f"{generations_folder}/", 
    }
    
    ### SETTINGS 
    input_types = [exp_gt, exp_text_kitml] 
    
    DEBUG = 0 # to test the evaluation script, only the firsts #{DEBUG} eleemnts with {DEBUG}!=0 are considered
    
    ###
    np.random.seed(0)
    device = "cpu"
    fps = 20.0
        
    names = [i["name"] for i in input_types]
    result = []

    if data_format=="Joints" and dataset=="kitml":
        testset_path = f"{os.getcwd()}/kit_numpy/test/"
        test_files = [f for f in os.listdir(testset_path) if f.endswith('_motion.npy')]
        ids_gt = [f.split('_')[0] for f in test_files]
    elif data_format=="Smpl":
        path_ids = f"{base_smpldataset_path}/annotations/{dataset}/splits/test.txt"
        with open(path_ids) as f:
            ids_gt = f.readlines()
            ids_gt = [i.strip() for i in ids_gt]
    else:
        raise NotImplementedError()

    if DEBUG != 0:
        ids_gt = ids_gt[:DEBUG]
    
    print(f"\nExperiment on {names} and with debug [{DEBUG}] of the split [{ids_gt}]\n")

    texts_gt, motions_guofeats_gt = [], []
    for idx in ids_gt:
        try:
            motion, text, length = load_input(idx, dataset=dataset, data_format=data_format, AMASS_path=f"{base_smpldataset_path}/motions/AMASS_20.0_fps_nh_smpljoints_neutral_nobetas/")
        except:
            continue

        texts_gt.append(text)
        if data_format=="Smpl":
            x, y, z = motion.T
            joint = np.stack((x, z, -y), axis=0).T
        else:
            joint = motion[0].view(motion.shape[1], 21, 3).cpu().numpy()
        feats = joints_to_guofeats(joint) # [num_frames, 263]
        motions_guofeats_gt.append(feats)
    
    tmr_forward = load_tmr_model_easy(device, dataset)
    
    print(f"texts_gt len {len(texts_gt)}")
    diversity_times = 300 if 300 < len(texts_gt) else len(texts_gt) - 1
    
    # texts_gt: list(str) - N elements for testset
    # motions_guofeats_gt: list: N * tensor(n_frames, 263)) - N elements for testset
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
            if "humanact12" in annotations[idx]["path"]:
                continue
            if experiment["name"] == "gt":
                motion_path = os.path.join(experiment["generations_folder"],annotations[idx]["path"]+".npy")
                motion = np.load(motion_path) # [num_frames, 24, 3], with 0 < num_frames   
                start_frame = int(annotations[idx]["annotations"][0]["start"] * fps)
                end_frame = int(annotations[idx]["annotations"][0]["end"] * fps)
                
                motion = motion[start_frame:end_frame] 
            else:
                motion_path = experiment["generations_folder"] + idx +".npy"
                motion = np.load(motion_path)
                if motion.shape[0] > 200:
                    motion = motion[:200]
        
            x, y, z = motion.T
            joint = np.stack((x, z, -y), axis=0).T # [num_frames, 24, 3]
            feats = joints_to_guofeats(joint) # [num_frames, 263]
            motions_guofeats.append(feats)
        
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
            inter_points = np.array([N // 4, 2 * N // 4, 3 * N // 4]) # tre frames 

            gt_motion_guofeats = torch.from_numpy(motion_guofeats) # (n_frames, 263)
            gt_joints_local = guofeats_to_joints_local(gt_motion_guofeats)
            
            gt_joints_local = gt_joints_local - gt_joints_local[:, [0]] # (n_frames, 22, 3) 

            # Same distance as in TEACH
            trans_dist_lst.append(
            torch.linalg.norm(
                    (gt_joints_local[inter_points] - gt_joints_local[inter_points - 1]), 
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
    
    
if __name__ == "__main__":
    main()