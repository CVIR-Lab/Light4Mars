# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class synthetic_rover(BaseSegDataset):
    METAINFO = dict(
        classes=('big rock', 'small rock', 'gravel', 'bedrock',
                 'ridge', 'sand', 'soil', 'sky'),
        palette=[[255, 255, 0], [255, 150, 0], [255, 0, 0], [0, 255, 0],
                 [255, 190, 255], [0, 255, 255], [0, 0, 255], [102, 102, 102]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
