#!/usr/bin/env python

import cv2
from core.detectors import CornerNet_Saccade
from core.vis_utils import draw_bboxes
from core.paths import get_file_path

import os

from pydagmtools import dagm
from pydagmtools import dagmjson

from core.dbs.dagm import DAGM

print("当前cwd: " + os.getcwd())
print("当前dir: " + os.path.abspath(os.path.dirname(__file__)))

# detector = CornerNet_Saccade()
# image = cv2.imread("demo.jpg")
#
# # cv2.imshow("asd", image)
# # cv2.waitKey()
#
# bboxes = detector(image)
# image = draw_bboxes(image, bboxes)
# cv2.imwrite("demo_out.jpg", image)


cfg_path = get_file_path("..", "configs", "CornerNet_Saccade.json")
print(cfg_path)
a = DAGM(cfg_path)

from core.base import Base, load_cfg, load_nnet
from core.paths import get_file_path
from core.config import SystemConfig
from core.dbs.coco import COCO

from core.test.cornernet import cornernet_inference
from core.models.CornerNet_Squeeze import model

cfg_path = get_file_path("..", "configs", "CornerNet_Squeeze.json")
model_path = get_file_path("..", "cache", "nnet", "CornerNet_Squeeze", "CornerNet_Squeeze_500000.pkl")
cfg_sys, cfg_db = load_cfg(cfg_path)
sys_cfg = SystemConfig().update_config(cfg_sys)
coco = COCO(cfg_db)

b = DAGM(cfg_path)

cornernet = load_nnet(sys_cfg, model())


