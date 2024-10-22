"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

# from ._dataset import DetDataset
from .cifar_dataset import CIFAR10
from .coco_dataset import CocoDetection
from .coco_dataset import (
    CocoDetection, 
    rm_category2name_v1, 
    rm_category2label_v1,
    rm_label2category_v1,
    rm_category2name_v2,
    rm_category2label_v2,
    rm_label2category_v2,
)
from .coco_eval import CocoEvaluator
from .coco_utils import get_coco_api_from_dataset
from .voc_detection import VOCDetection
from .voc_eval import VOCEvaluator
