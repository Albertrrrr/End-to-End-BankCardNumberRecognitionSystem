#-*- coding:utf-8 -*-
import os
import demoSupportOCR
import time
import shutil
import numpy as np
from PIL import Image
from glob import glob
import re

def indexFirst(result):
    dictKeys = []
    for keys in result:
        dictKeys.append(keys)
    firstIndex = dictKeys[0:1][0]

    return firstIndex


def findMaxBox(result):
    result_list = list(result.values())[1:]
    local = []
    for i in result_list:
        local.append(list(i)[0])
    length = []
    for i in local:
        local_every_list = list(i)
        res = local_every_list[0:3:2]
        now = res[1] - res[0]
        length.append(now)
    res_length = length
    if(indexFirst(result) == 1):
        max_index = res_length.index(max(length)) + 2
    if(indexFirst(result) == 0):
        max_index = res_length.index(max(length)) + 1
        if(max_index>=9):
            max_index = res_length.index(max(length)) + 2
    return max_index


def output(result):
    max_index = findMaxBox(result)
    return result[max_index][1]


def normalization(Str):
    modelStr = "\d+"
    res_str = "".join(re.findall(modelStr,Str))
    if(res_str[0:1] == '1'):
        res_str = res_str[1:]
    return res_str

def normalization_other(Str):
    modelStr = "\d+"
    res_str = "".join(re.findall(modelStr, Str))
    return res_str

def check(result):
    res_str = normalization(output(result))
    if(len(res_str)<16):
        max_index = findMaxBox(result)
        max_point = result[max_index][0][2]
        result_list = list(result.values())
        local = []
        for i in result_list:
            local.append(list(i)[0])
        length_max = []
        length=[]
        for i in local:
            local_every_list = list(i)
            res = local_every_list[2]
            length.append(res)
            if(res>max_point):
                length_max.append(res)
        index = length.index(min(length_max))
        next_res_str = result[index][1]
        return res_str + normalization_other(next_res_str)
    else:
        return res_str


if __name__ == '__main__':
    image_files = glob(os.getcwd() + r"/test_images/*.*")
    result_dir = os.getcwd() + r"/test_result"
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.mkdir(result_dir)

    for image_file in sorted(image_files):
        image = np.array(Image.open(image_file).convert('RGB'))
        t = time.time()

        #ctpn检测位置后进行调入
        result, image_framed = demoSupportOCR.model(image)
        output_file = os.path.join(result_dir, image_file.split('/')[-1])
        Image.fromarray(image_framed).save(output_file)

        #输出时间
        print("Mission complete, it took {:.3f}s".format(time.time() - t))
        print("Bank Card Recognition Result:")
        # for key in result:
        #     print(result[key][1])
        # print(output(result))

        for keys in result:
            a = result[keys][0]
            result[keys][0] = a.tolist()
        print(check(result))

#GUI接口
def loadSystem(url):
    image_url = str(url)
    image_files = glob(image_url)
    result_dir = './test_result'
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.mkdir(result_dir)
    for image_file in sorted(image_files):
         image= np.array(Image.open(image_file).convert('RGB'))
         # t = time.time()

         #ctpn检测位置后进行调入
         result, image_framed = demoSupportOCR.model(image)
         output_file = os.path.join(result_dir, image_file.split('/')[-1])
         Image.fromarray(image_framed).save(output_file)



         for keys in result:
            a = result[keys][0]
            result[keys][0] = a.tolist()

    return check(result)


