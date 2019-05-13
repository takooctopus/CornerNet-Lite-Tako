import os
import cv2
import imghdr
import glob as gb

import time
import datetime
import json

from pydagmtools import dagm
from pydagmtools import dagmjson

print(os.getcwd())
print(os.path.abspath(os.path.dirname(__file__)))

a = dagm.DAGM('../annotations/testdev2017foo.json')
j = dagmjson.DAGMjson()

json_configs = j.jsonConfigs
img_index = 0

for ind_cat, cat in enumerate(j.cats):
    configs = j.configs
    category_config = configs['category']
    category_config.update({
        "id": ind_cat,
        "name": cat,
    })
    if category_config not in json_configs['categories']:
        json_configs['categories'].append(category_config)

    trainDir = j.dataDir + cat + '/Train/'
    trainLabelFile = trainDir + '/Label/'
    testDir = j.dataDir + cat + '/Test/'
    trainLabelFile = testDir + '/Label/'
    train_img_path = gb.glob(trainDir + "*.PNG")
    train_label_path = gb.glob(trainLabelFile + "*.PNG")
    for ind_img, path in enumerate(train_img_path):
        img_index += 1
        image_config = configs['image']

        img = cv2.imread(path)
        shape = img.shape
        # filePath = unicode(path, 'utf8')
        timestamp = os.path.getmtime(path)
        timeStruct = time.localtime(timestamp)
        file_create_time = time.strftime('%Y-%m-%d %H:%M:%S', timeStruct)
        image_config.update({
            'file_name': path,
            'height': shape[0],
            'weight': shape[1],
            'date_captured': file_create_time,
            'id': img_index
        })
        if image_config not in json_configs['images']:
            json_configs['images'].append(image_config)
        print(json.dumps(image_config, sort_keys=True, indent=4))
        # break
    # break

print(json.dumps(json_configs, sort_keys=True, indent=4))
# print(json.dumps(image, sort_keys=True, indent=4))
