_base_ = ['../_base_/models/lmars.py', '../_base_/datasets/synrover.py',
          '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py']
checkpoint = '../../pretrain/light4mars.pth'
crop_size = (512, 512)

data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        type='lmars',
        c_list=[32, 64, 128, 256],
        num_heads=(1, 2, 4, 8),
        num_layers=[2, 2, 2, 2],
    ),
    decode_head=dict(
        type='lmarshead',
        in_channels=[32, 64, 128, 256],
        num_classes=8))
train_cfg = dict(type='IterBasedTrainLoop', max_iters=160000, val_interval=2500)
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=20000),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook')
)

train_dataloader = dict(batch_size=18, num_workers=2)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0002, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]
