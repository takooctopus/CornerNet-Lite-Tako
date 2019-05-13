#!/usr/bin/env python

import os
import cv2
import imghdr
import glob as gb

import time
import datetime
import json
import re
import numpy as np

from pydagmtools import dagm
from pydagmtools import dagmjson

print(os.getcwd())
print(os.path.abspath(os.path.dirname(__file__)))

a = dagm.DAGM('../annotations/testdev2017foo.json')
j = dagmjson.DAGMjson()

json_configs = j.jsonConfigs
img_index = 0
label_index = 0
img_search_index = 0

for ind_cat, cat in enumerate(j.cats):
    print(str(ind_cat+1) + " : " + cat)
    configs = j.configs
    category_config = configs['category']
    category_config.update({
        "id": ind_cat + 1,
        "name": cat,
    })
    if category_config not in json_configs['categories']:
        json_configs['categories'].append(category_config.copy())

    imageDir = j.dataDir + cat + '/'
    labelDir = imageDir + 'Label/'
    img_path = gb.glob(imageDir + "*.PNG")
    label_path = gb.glob(labelDir + "*.PNG")
    for ind_img, path in enumerate(img_path):
        img_index += 1
        image_config = configs['image']
        img = cv2.imread(path)
        shape = img.shape
        # filePath = unicode(path, 'utf8')
        timestamp = os.path.getmtime(path)
        timeStruct = time.localtime(timestamp)
        file_create_time = time.strftime('%Y-%m-%d %H:%M:%S', timeStruct)
        image_config.update({
            'file_name': os.path.basename(path),
            'dagm_url': path,
            'height': shape[0],
            'width': shape[1],
            'date_captured': file_create_time,
            'id': img_index
        })
        if image_config not in json_configs['images']:
            json_configs['images'].append(image_config.copy())
        # print(json.dumps(image_config, sort_keys=True, indent=4))
        # break

    for ind_img, path in enumerate(label_path):
        label_index += 1
        annotation_config = configs['annotation']
        label = cv2.imread(path)
        label_name = os.path.basename(path)
        img_name = re.sub(r'_label', "", label_name)
        img_id = None
        for i in range(img_search_index, len(json_configs['images'])):
            if json_configs['images'][i]['file_name'] == img_name:
                img_id = json_configs['images'][i]['id']
                # print(i)
                # print(img_id)
                # print(json_configs['images'][i]['file_name'])
                # print(img_name)
                break
        img_search_index =+ len(json_configs['images'])
        img_search_index =- 1
        assert img_id is not None

        gray = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        segmentation = contours[0].reshape(-1).tolist()
        x, y, w, h = cv2.boundingRect(contours[0])
        x = x + w / 2
        y = y + h / 2
        w = float(w)
        h = float(h)
        area = float(w * h)

        # cv2.rectangle(label, (x, y), (x + w, y + h), (0, 0, 200), 2)
        #
        # while (1):
        #     cv2.imshow('img', label)
        #     cv2.imshow('gray', gray)
        #     cv2.imshow('blur', blur)
        #     cv2.imshow('thresh', thresh)
        #     if cv2.waitKey(1) == ord('q'):
        #         break
        # cv2.destroyAllWindows()

        annotation_config.update({
            "id": label_index,
            "image_id": img_id,
            "category_id": ind_cat + 1,
            "segmentation": segmentation,
            "area": area,
            "bbox": [x, y, w, h],
        })
        if annotation_config not in json_configs['annotations']:
            json_configs['annotations'].append(annotation_config.copy())
        # break
    # break

with open('demo.json', 'w') as f:
    json.dump(json_configs, f)

# print(json.dumps(json_configs, sort_keys=True, indent=4))
# print(json.dumps(image, sort_keys=True, indent=4))
