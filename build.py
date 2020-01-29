# -*- coding: utf-8 -*-
'''
教師データをビルドするプログラム
　- config/class.yamlに各ビルド設定を記載
　- ビルド後の画像
'''

import os
import sys
import shutil
import glob
import yaml
from tqdm import tqdm


if len(sys.argv) < 2:
    print('plese enter classs name!!')
    exit(0)


# 登録済みのクラス一覧
classes = yaml.load(open('config/class.yaml'), Loader=yaml.SafeLoader)
class_name = sys.argv[1]
if classes == None:
    classes = []
if not class_name in classes:
    classes.append(sys.argv[1])
    yaml.dump(classes, open('config/class.yaml', 'w'))
os.makedirs('data/' + class_name, exist_ok=True)

# tmp/配下の画像データ
files = glob.glob('tmp/*')


# ファイルを移動
for file in tqdm(files):
    shutil.move(file, 'data/' + class_name)
