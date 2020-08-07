# coding: utf-8
import numpy as np
import glob
import os
import json

rawdata_name = 'GGC-raw-data'  # 読み込むrawデータの名前を指定
path_list = glob.glob(os.path.join('GGC-pictures', '*'))
data_dict = {}

with open(os.path.join('raw_data', rawdata_name + '.txt')) as f:
    while True:
        filename = f.readline().rstrip()
        image_data = []
        for i in range(224):
            image_row = []
            for j in range(224):
                row = f.readline()[4:51]
                x_1 = row[:15]
                x_2 = row[16:31]
                x_3 = row[32:]
                pixel_data = list(map(float, [x_1, x_2, x_3]))
                image_row.append(pixel_data)
            
            new_line = f.readline()
            if  new_line not in ('\n', 'end\n'):
                print('new line error')
                print(new_line)
                Exception
            image_data.append(image_row)
        
        new_line = f.readline()
        if new_line != '\n':
            print('end new line error')
            print(new_line)
            Exception
        
        data_dict[filename] = image_data
        if len(data_dict) == len(path_list):
            break

s1 = set([os.path.split(path)[-1] for path in path_list])
s2 = set(data_dict.keys())

if len(s1 - s2) != 0:
    print(s1 - s2)
    Exception

if not os.path.isdir('GuidedGradCAM_data'):
    os.mkdir('GuidedGradCAM_data')
with open(os.path.join('GuidedGradCAM_data', rawdata_name + '.json', 'w') as f:
    json.dump(data_dict, f)




