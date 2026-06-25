_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
#batch_size = 12  # bs: total bs in all gpus
batch_size = 3 # my
#num_worker = 24
num_worker = 1
mix_prob = 0.8
#empty_cache = False
empty_cache = True
#enable_amp = True   
enable_amp = False
eval_epoch = 3
epoch = 9 #my

enable_wandb = True
wandb_offline=True  # True=离线, False=在线
wandb_project = "FDPTV3" # my
wandb_key = "df9713a6e19f7a24bbb40640a2cf4615345f6961" # my

train = dict(type="FedTrainer")

# ############################ 联邦学习相关设置 ###############################
federated = dict(
    # --- 核心参数 ---
    num_users=3,
    total_rounds=100,
    users_per_gpu=1,
    msg="FedAvg",
    
    # --- 算法选择 ---
    # 可选项: 'FedAvg', 'FedAvgM', 'FedProx', 'FedAdam', 'FedMarkovAvg'
    aggregation_method="FedAvg",

    # --- 数据拆分策略配置（使用注册器机制）---
    data_split_strategy=dict(
        s3disdataset=dict(
            type="S3DISSplitter",
            areas=("Area_1", "Area_2", "Area_3", "Area_4", "Area_6"),
            validation_area="Area_5",
        ),
    ),

    client=dict(
        type="FedClientBase",  # 使用基础客户端
        weight_mode="standard",  # "standard"(默认逐层数组) | "structured"(二进制打包)
    ),

    # --- 算法专属超参数 ---
    hyperparameters=dict(
        fedavg=dict(),
        fedavgm=dict(
            beta=0.9,
            server_lr=1.0,
            server_lr_scheduler=dict(
                type="FedServerLinearWarmupLR",
                initial_lr=0.7,
                warmup_rounds=30,
                warmup_start_lr=0.7,
                max_lr=1.2,
                final_lr=1.0
            ),
            server_momentum_scheduler=dict(
                type="FedServerLinearWarmupMomentum",
                initial_beta=0.3,
                warmup_rounds=30,
                warmup_start_beta=0.3,
                max_beta=0.90,
                final_beta=0.90
            ),
        ),
        fedprox=dict(
            mu=0.01
        ),
        fedadam=dict(
            lr=0.001,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            weight_decay=0.0,
            server_lr_scheduler=dict(
                type="FedServerCosineAnnealingLR",
                initial_lr=0.001,
                min_lr=0.0001
            )
        ),
        fedmarkovavg=dict(
            epsilon=1e-6  # 服务端也增加数值稳定性
        )
    )
)
# ##############################################

# model settings
model = dict(
    type="DefaultSegmentorV2",
    num_classes=13,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m1",
        in_channels=6,
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(128, 128, 128, 128, 128),   #
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(128, 128, 128, 128),    # 
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        ############################################
        #enable_rpe=True,
        # enable_flash=False,     #开启flash attention，要关闭upcast_attention和upcast_softmax和enable_amp
        # upcast_attention=True,  #反之关闭flash attention要开启upcast_attention和upcast_softmax和enable_amp 
        # upcast_softmax=True,
        enable_rpe= False,
        enable_flash=True,    
        upcast_attention=False,  
        upcast_softmax=False,
        ############################################
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# scheduler settings
#epoch = 3000 # my
optimizer = dict(type="AdamW", lr=0.006, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.006, 0.0006],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0006)]

# dataset settings
dataset_type = "S3DISDataset"
data_root = "data/s3dis_normal"

data = dict(
    num_classes=13,
    ignore_index=-1,
    names=[
        "ceiling",
        "floor",
        "wall",
        "beam",
        "column",
        "window",
        "door",
        "table",
        "chair",
        "sofa",
        "bookcase",
        "board",
        "clutter",
    ],
    train=dict(
        type=dataset_type,
        split=("Area_1", "Area_2", "Area_3", "Area_4", "Area_6"),
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
            ),
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
            # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
            dict(
                type="GridSample",
                #grid_size=0.02,    
                grid_size=0.02,  # my
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="SphereCrop", sample_rate=0.6, mode="random"),
            dict(type="SphereCrop", point_max=204800, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            # dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("color", "normal"),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="Area_5",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                #grid_size=0.02,    
                grid_size=0.02,  # my
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"),
                feat_keys=("color", "normal"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="Area_5",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                #grid_size=0.02,
                grid_size=0.02,  # my
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("color", "normal"),
                ),
            ],
            aug_transform=[
                [dict(type="RandomScale", scale=[0.9, 0.9])],
                [dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomScale", scale=[1, 1])],
                [dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomScale", scale=[1.1, 1.1])],
                [
                    dict(type="RandomScale", scale=[0.9, 0.9]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[1, 1]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[1.1, 1.1]),
                    dict(type="RandomFlip", p=1),
                ],
            ],
        ),
    ),
)
