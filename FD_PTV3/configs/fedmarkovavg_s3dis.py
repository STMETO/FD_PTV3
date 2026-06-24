"""
FD_PTV3 FedMarkovAvg S3DIS 配置示例
====================================
使用马尔科夫联邦平均算法 + 结构化权重通信。

用法:
    python -m FD_PTV3.fd_train --config-file FD_PTV3/configs/fedmarkovavg_s3dis.py
"""

weight = None
resume = False
evaluate = True
test_only = False
seed = 27567705
save_path = 'exp/s3dis/fd_fedmarkovavg_test'
num_worker = 1
batch_size = 3
gradient_accumulation_steps = 1
batch_size_val = None
batch_size_test = None
eval_epoch = 3
clip_grad = 1.0
sync_bn = False
enable_amp = False
amp_dtype = 'float16'
empty_cache = True
empty_cache_per_epoch = False
find_unused_parameters = False
enable_wandb = True
wandb_project = 'FD_PTV3'
wandb_key = ''
mix_prob = 0.8
param_dicts = [dict(keyword='block', lr=0.0006)]

hooks = [
    dict(type='CheckpointLoader'),
    dict(type='ModelHook'),
    dict(type='IterationTimer', warmup_iter=2),
    dict(type='InformationWriter'),
    dict(type='SemSegEvaluator'),
    dict(type='CheckpointSaver', save_freq=None),
    dict(type='PreciseEvaluator', test_last=False),
]

train = dict(type='FedTrainer')
test = dict(type='SemSegTester', verbose=True)
epoch = 9

# ================================================================
# 联邦学习配置 (FedMarkovAvg)
# ================================================================
federated = dict(
    num_users=3,
    total_rounds=100,
    users_per_gpu=1,
    msg='FD_PTV3 Flower FedMarkovAvg on S3DIS',
    aggregation_method='FedMarkovAvg',

    client=dict(
        type='MarkovFedClient',
        aggre_mode='FedMarkovAvg',
        binarize_all_layers=True,
        verbose=False,
    ),

    hyperparameters=dict(
        fedmarkovavg=dict(
            aggre_mode='FedMarkovAvg',
            epsilon=1e-8,
            EDE=False,
            global_epochs=100,
        ),
    ),

    data_split_strategy=dict(
        S3DISDataset=dict(
            type='S3DISSplitter',
            areas=('Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6'),
        ),
    ),
)

# ================================================================
# 模型配置
# ================================================================
model = dict(
    type='DefaultSegmentorV2',
    num_classes=13,
    backbone_out_channels=64,
    backbone=dict(
        type='PT-v3m1',
        in_channels=6,
        order=['z', 'z-trans', 'hilbert', 'hilbert-trans'],
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(128, 128, 128, 128, 128),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(128, 128, 128, 128),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=('ScanNet', 'S3DIS', 'Structured3D'),
    ),
    criteria=[
        dict(type='CrossEntropyLoss', loss_weight=1.0, ignore_index=-1),
        dict(type='LovaszLoss', mode='multiclass', loss_weight=1.0, ignore_index=-1),
    ],
)

optimizer = dict(type='AdamW', lr=0.006, weight_decay=0.05)
scheduler = dict(
    type='OneCycleLR',
    max_lr=[0.006, 0.0006],
    pct_start=0.05,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=1000.0,
)

dataset_type = 'S3DISDataset'
data_root = 'data/s3dis_normal'

data = dict(
    num_classes=13,
    ignore_index=-1,
    names=[
        'ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
        'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter',
    ],
    train=dict(
        type='S3DISDataset',
        split=('Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6'),
        data_root='data/s3dis_normal',
        transform=[
            dict(type='CenterShift', apply_z=True),
            dict(type='RandomDropout', dropout_ratio=0.2, dropout_application_ratio=0.2),
            dict(type='RandomRotate', angle=[-1, 1], axis='z', center=[0, 0, 0], p=0.5),
            dict(type='RandomRotate', angle=[-0.015625, 0.015625], axis='x', p=0.5),
            dict(type='RandomRotate', angle=[-0.015625, 0.015625], axis='y', p=0.5),
            dict(type='RandomScale', scale=[0.9, 1.1]),
            dict(type='RandomFlip', p=0.5),
            dict(type='RandomJitter', sigma=0.005, clip=0.02),
            dict(type='ChromaticAutoContrast', p=0.2, blend_factor=None),
            dict(type='ChromaticTranslation', p=0.95, ratio=0.05),
            dict(type='ChromaticJitter', p=0.95, std=0.05),
            dict(type='GridSample', grid_size=0.02, hash_type='fnv', mode='train', return_grid_coord=True),
            dict(type='SphereCrop', sample_rate=0.6, mode='random'),
            dict(type='SphereCrop', point_max=204800, mode='random'),
            dict(type='CenterShift', apply_z=False),
            dict(type='NormalizeColor'),
            dict(type='ToTensor'),
            dict(type='Collect', keys=('coord', 'grid_coord', 'segment'), feat_keys=('color', 'normal')),
        ],
        test_mode=False,
        loop=3,
    ),
    val=dict(
        type='S3DISDataset',
        split='Area_5',
        data_root='data/s3dis_normal',
        transform=[
            dict(type='CenterShift', apply_z=True),
            dict(type='Copy', keys_dict=dict(segment='origin_segment')),
            dict(type='GridSample', grid_size=0.02, hash_type='fnv', mode='train',
                 return_grid_coord=True, return_inverse=True),
            dict(type='CenterShift', apply_z=False),
            dict(type='NormalizeColor'),
            dict(type='ToTensor'),
            dict(type='Collect', keys=('coord', 'grid_coord', 'segment', 'origin_segment', 'inverse'),
                 feat_keys=('color', 'normal')),
        ],
        test_mode=False,
    ),
    test=dict(
        type='S3DISDataset',
        split='Area_5',
        data_root='data/s3dis_normal',
        transform=[
            dict(type='CenterShift', apply_z=True),
            dict(type='NormalizeColor'),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(type='GridSample', grid_size=0.02, hash_type='fnv', mode='test',
                          return_grid_coord=True),
            crop=None,
            post_transform=[
                dict(type='CenterShift', apply_z=False),
                dict(type='ToTensor'),
                dict(type='Collect', keys=('coord', 'grid_coord', 'index'),
                     feat_keys=('color', 'normal')),
            ],
            aug_transform=[
                [{'type': 'RandomScale', 'scale': [0.9, 0.9]}],
                [{'type': 'RandomScale', 'scale': [0.95, 0.95]}],
                [{'type': 'RandomScale', 'scale': [1, 1]}],
                [{'type': 'RandomScale', 'scale': [1.05, 1.05]}],
                [{'type': 'RandomScale', 'scale': [1.1, 1.1]}],
                [{'type': 'RandomScale', 'scale': [0.9, 0.9]}, {'type': 'RandomFlip', 'p': 1}],
                [{'type': 'RandomScale', 'scale': [0.95, 0.95]}, {'type': 'RandomFlip', 'p': 1}],
                [{'type': 'RandomScale', 'scale': [1, 1]}, {'type': 'RandomFlip', 'p': 1}],
                [{'type': 'RandomScale', 'scale': [1.05, 1.05]}, {'type': 'RandomFlip', 'p': 1}],
                [{'type': 'RandomScale', 'scale': [1.1, 1.1]}, {'type': 'RandomFlip', 'p': 1}],
            ],
        ),
    ),
)
