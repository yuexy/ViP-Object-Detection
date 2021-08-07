_base_ = './vip_s_cascade_mask_rcnn_1x.py'

lr_config = dict(step=[27, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)
