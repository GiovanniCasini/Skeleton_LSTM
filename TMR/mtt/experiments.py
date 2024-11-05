from pathlib import Path

MDM_FOLDER = Path("/home/mathis/motion-diffusion-model/save/humanml_trans_enc_512")
MDM_SMPL_FOLDER = Path(
    "/andromeda/personal/lmandelli/stmc/pretrained_models/mdm-smpl_clip_smplrifke_humanml3d"
)
MotionDiffuse_FOLDER = Path("/home/mathis/stmc/motion_diffuse")

mdm_experiments = [
    {
        "folder": MDM_FOLDER
        / "samples_humanml_trans_enc_512_000200000_seed10_mtt_timeline_baseline_onetext",
        "name": r"MDM One Text",
        "y_is_z_axis": True,
        "only_text": True,
    },
    {
        # MDM STMC NEW
        "folder": MDM_FOLDER
        / "samples_humanml_trans_enc_512_000200000_seed10_mtt_timeline_baseline_singletrack",
        "name": r"MDM DiffCollage",
        "y_is_z_axis": True,
    },
    {
        # MDM STMC NEW
        "folder": MDM_FOLDER
        / "samples_humanml_trans_enc_512_000200000_seed10_mtt_timeline_baseline_sinc",
        "name": r"MDM SINC",
        "y_is_z_axis": True,
    },
    {
        # MDM STMC NEW
        "folder": MDM_FOLDER
        / "samples_humanml_trans_enc_512_000200000_seed10_mtt_timeline_baseline_sinc_lerp",
        "name": r"MDM SINC lerp",
        "y_is_z_axis": True,
    },
    {
        # MDM STMC NEW
        "folder": MDM_FOLDER
        / "samples_humanml_trans_enc_512_000200000_seed10_MTT.txt_timeline",
        "name": r"MDM STMC NEW",
        "y_is_z_axis": True,
    },
    {
        # MDM STMC OLD
        "folder": MDM_FOLDER
        / "samples_humanml_trans_enc_512_000200000_seed10_STMotions_v4_interval",
        "name": r"MDM STMC OLD",
        "y_is_z_axis": True,
        "skip": True,
    },
]

mdm_smpl_experiments = [
    {
        # NEW, baseline in the code joints
        "folder": MDM_SMPL_FOLDER / "mtt_generations_last_from_joints_baseline_onetext",
        "name": r"MS One text",
        "only_text": True,
    },
    {
        # NEW, baseline in the code joints
        "folder": MDM_SMPL_FOLDER
        / "mtt_generations_last_from_joints_baseline_singletrack",
        "name": r"MS DiffCollage",
    },
    {
        # NEW, baseline in the code joints
        "folder": MDM_SMPL_FOLDER / "mtt_generations_last_from_joints_baseline_sinc",
        "name": r"MS SINC",
    },
    {
        # NEW, baseline in the code joints
        "folder": MDM_SMPL_FOLDER
        / "mtt_generations_last_from_joints_baseline_sinc_lerp",
        "name": r"MS SINC lerp",
    },
    {
        # NEW, in the code joints
        "folder": MDM_SMPL_FOLDER / "mtt_generations_last_from_joints",
        "name": r"MS STMC",
    },
]


motion_diffuse_experiments = [
    {
        "folder": MotionDiffuse_FOLDER
        / "t2m_motiondiffuse_mtt_timeline_baseline_onetext",
        "name": r"MotionDiffuse One Text",
        "y_is_z_axis": True,
        "only_text": True,
    },
    {
        "folder": MotionDiffuse_FOLDER
        / "t2m_motiondiffuse_mtt_timeline_baseline_singletrack",
        "name": r"MotionDiffuse DiffCollage",
        "y_is_z_axis": True,
    },
    {
        "folder": MotionDiffuse_FOLDER / "t2m_motiondiffuse_mtt_timeline_baseline_sinc",
        "name": r"MotionDiffuse SINC",
        "y_is_z_axis": True,
    },
    {
        "folder": MotionDiffuse_FOLDER
        / "t2m_motiondiffuse_mtt_timeline_baseline_sinc_lerp",
        "name": r"MotionDiffuse SINC lerp",
        "y_is_z_axis": True,
    },
    {
        "folder": MotionDiffuse_FOLDER / "t2m_motiondiffuse_mtt_timeline",
        "name": r"MotionDiffuse STMC",
        "y_is_z_axis": True,
    },
]


experiments =  mdm_smpl_experiments + mdm_experiments + motion_diffuse_experiments


mdm_smpl_experiments_s = [
    {
        # NEW, baseline in the code joints
        "folder": MDM_SMPL_FOLDER / "mtt_generations_last_from_smpl_baseline_onetext",
        "name": r"MS One text",
        "only_text": True,
    },
    {
        # NEW, baseline in the code joints
        "folder": MDM_SMPL_FOLDER
        / "mtt_generations_last_from_smpl_baseline_singletrack",
        "name": r"MS DiffCollage",
    },
    {
        # NEW, baseline in the code joints
        "folder": MDM_SMPL_FOLDER / "mtt_generations_last_from_smpl_baseline_sinc",
        "name": r"MS SINC",
    },
    {
        # NEW, baseline in the code joints
        "folder": MDM_SMPL_FOLDER / "mtt_generations_last_from_smpl_baseline_sinc_lerp",
        "name": r"MS SINC lerp",
    },
    {
        # NEW, in the code joints
        "folder": MDM_SMPL_FOLDER / "mtt_generations_last_from_smpl",
        "name": r"MS STMC",
    },
]

experiments = mdm_smpl_experiments_s

sup_mat_exp = [
    {
        # NEW, in the code joints
        "folder": MDM_SMPL_FOLDER
        / "mtt_generations_last_from_joints_intervaloverlap_2.0",
        "name": r"MS STMC 2.0",
    },
    {
        # NEW, in the code joints
        "folder": MDM_SMPL_FOLDER
        / "mtt_generations_last_from_joints_intervaloverlap_0.25",
        "name": r"MS STMC 0.25",
    },
    {
        # NEW, in the code joints
        "folder": MDM_SMPL_FOLDER
        / "mtt_generations_last_from_joints_intervaloverlap_0.5",
        "name": r"MS STMC 0.5",
    },
    {
        # NEW, in the code joints
        "folder": MDM_SMPL_FOLDER
        / "mtt_generations_last_from_joints_intervaloverlap_0.75",
        "name": r"MS STMC 0.75",
    },
    {
        # NEW, in the code joints
        "folder": MDM_SMPL_FOLDER / "mtt_generations_last_from_joints",
        "name": r"MS STMC 1.0",
    },
    {
        # NEW, in the code joints
        "folder": MDM_SMPL_FOLDER
        / "mtt_generations_last_from_joints_intervaloverlap_1.25",
        "name": r"MS STMC 1.25",
    },
    {
        # NEW, in the code joints
        "folder": MDM_SMPL_FOLDER
        / "mtt_generations_last_from_joints_intervaloverlap_1.5",
        "name": r"MS STMC 1.5",
    },
    {
        # NEW, in the code joints
        "folder": MDM_SMPL_FOLDER
        / "mtt_generations_last_from_joints_intervaloverlap_1.75",
        "name": r"MS STMC 1.75",
        "skip": True,
    },
    {
        # NEW, in the code joints
        "folder": MDM_SMPL_FOLDER
        / "mtt_generations_last_from_joints_intervaloverlap_2.0",
        "name": r"MS STMC 2.0",
    },
]

experiments = sup_mat_exp

new_look_mdm_smpl = [
    {
        # NEW, baseline in the code joints
        "folder": MDM_SMPL_FOLDER / "mtt_generations_last_from_joints_baseline_onetext",
        "name": r"MS One text",
        "only_text": True,
        "skip": True,
    },
    {
        # NEW, baseline in the code joints
        "folder": MDM_SMPL_FOLDER
        / "mtt_generations_last_from_joints_baseline_singletrack_intervaloverlap_0.5",
        "name": r"MS DiffCollage 0.5",
    },
    {
        # NEW, baseline in the code joints
        "folder": MDM_SMPL_FOLDER
        / "mtt_generations_last_from_joints_baseline_sinc_intervaloverlap_0.5",
        "name": r"MS SINC 0.5",
    },
    {
        # NEW, baseline in the code joints
        "folder": MDM_SMPL_FOLDER
        / "mtt_generations_last_from_joints_baseline_sinc_lerp_intervaloverlap_0.5",
        "name": r"MS SINC lerp 0.5",
    },
    {
        # NEW, in the code joints
        "folder": MDM_SMPL_FOLDER
        / "mtt_generations_last_from_joints_intervaloverlap_0.5",
        "name": r"MS STMC 0.5",
    },
]

# motiondiffuse
new_look = [
    {
        "folder": MotionDiffuse_FOLDER
        / "t2m_motiondiffuse_mtt_timeline_baseline_singletrack",
        "name": r"MotionDiffuse DiffCollage",
        "y_is_z_axis": True,
    },
    {
        "folder": MotionDiffuse_FOLDER
        / "t2m_motiondiffuse_mtt_timeline_baseline_sinc_lerp_intervaloverlap_0.5",
        "name": r"MotionDiffuse SINC lerp",
        "y_is_z_axis": True,
    },
    {
        "folder": MotionDiffuse_FOLDER / "t2m_motiondiffuse_mtt_timeline",
        "name": r"MotionDiffuse STMC",
        "y_is_z_axis": True,
    },
]


new_look = [
    {
        # MDM STMC NEW
        "folder": MDM_FOLDER
        / "samples_humanml_trans_enc_512_000200000_seed10_mtt_timeline_baseline_sinc_lerp_intervaloverlap_0.5",
        "name": r"MDM SINC lerp 0.5",
        "y_is_z_axis": True,
    },
    {
        "folder": MotionDiffuse_FOLDER
        / "t2m_motiondiffuse_mtt_timeline_baseline_sinc_lerp_intervaloverlap_0.5",
        "name": r"MotionDiffuse SINC lerp 0.5",
        "y_is_z_axis": True,
    },
]

_new_look = [
    {
        "folder": MotionDiffuse_FOLDER
        / "t2m_motiondiffuse_mtt_timeline_intervaloverlap_0.5",
        "name": r"MotionDiffuse STMC 0.5",
        "y_is_z_axis": True,
    }
]

new_look = [
    {
        # MDM STMC NEW
        "folder": MDM_FOLDER
        / "samples_humanml_trans_enc_512_000200000_seed10_mtt_timeline_intervaloverlap_0.5",
        "name": r"MDM STMC 0.5",
        "y_is_z_axis": True,
    }
]


new_look = [
    {
        "folder": MotionDiffuse_FOLDER
        / "t2m_motiondiffuse_mtt_timeline_baseline_singletrack_intervaloverlap_0.5",
        "name": r"MotionDiffuse DiffCollage 0.5",
        "y_is_z_axis": True,
    }
]

new_look = [
    {
        # MDM STMC NEW
        "folder": MDM_FOLDER
        / "samples_humanml_trans_enc_512_000200000_seed10_mtt_timeline_baseline_singletrack_intervaloverlap_0.5",
        "name": r"MDM DiffCollage 0.5",
        "y_is_z_axis": True,
    }
]

sup_mat_exp = [
    {
        # NEW, in the code joints
        "folder": MDM_SMPL_FOLDER
        / "mtt_generations_last_from_joints_intervaloverlap_0.4",
        "name": r"MS STMC 0.4",
    },
    {
        # NEW, in the code joints
        "folder": MDM_SMPL_FOLDER
        / "mtt_generations_last_from_joints_intervaloverlap_0.6",
        "name": r"MS STMC 0.6",
    },
]

# experiments = sup_mat_exp

experiments = new_look

mdm_smpl_experiments_changing_w = [
    {
        # NEW, in the code joints
        "folder": MDM_SMPL_FOLDER / "mtt_generations_last_from_joints_w11",
        "name": r"MS STMC",
    },
    {
        # NEW, in the code joints
        "folder": MDM_SMPL_FOLDER / "mtt_generations_last_from_joints_w12",
        "name": r"MS STMC",
    },
    {
        # NEW, in the code joints
        "folder": MDM_SMPL_FOLDER / "mtt_generations_last_from_joints_w13",
        "name": r"MS STMC",
    },
    {
        # NEW, in the code joints
        "folder": MDM_SMPL_FOLDER / "mtt_generations_last_from_joints_w14",
        "name": r"MS STMC",
    },
    {
        # NEW, in the code joints
        "folder": MDM_SMPL_FOLDER / "mtt_generations_last_from_joints_w16",
        "name": r"MS STMC",
    },
    {
        # NEW, in the code joints
        "folder": MDM_SMPL_FOLDER / "mtt_generations_last_from_joints_w17",
        "name": r"MS STMC",
    },
    {
        # NEW, in the code joints
        "folder": MDM_SMPL_FOLDER / "mtt_generations_last_from_joints_w18",
        "name": r"MS STMC",
    },
    {
        # NEW, in the code joints
        "folder": MDM_SMPL_FOLDER / "mtt_generations_last_from_joints_w19",
        "name": r"MS STMC",
    }
]

mdm_smpl_experiments_just_mdmsmpl = [
    {
        # NEW, in the code joints
        "folder": MDM_SMPL_FOLDER / "mtt_generations_last_from_joints",
        "name": r"MS STMC",
    }
]

As = [5, 6, 7, 8, 9]
import itertools
# Usa itertools.product per ottenere tutte le combinazioni possibili
combinazioni = list(itertools.product(As, repeat=3))

mdm_smpl_experiments_comb_w1w2w3 = [
    {
        "folder": MDM_SMPL_FOLDER / f"mtt_generations_last_from_joints_B2_wA{c[0]}_wB{c[1]}_wC{c[2]}",
         "name": r"MS STMC - " + f"A{c[0]}_B{c[1]}_C{c[2]}",
    } for  c  in combinazioni
]

mdm_smpl_experiment_badabim_2 = [
    {
        "folder": MDM_SMPL_FOLDER / f"mtt_generations_last_from_joints_B2_wA5_wB5_wC5",
        "name": r"MS STMC - " + f"B2_wA5_wB7.5_wC5",
    } 
]

experiments =  mdm_smpl_experiments_just_mdmsmpl