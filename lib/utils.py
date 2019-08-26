
import os
import re
import glob
import trimesh
import json
import sys
import math
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import exposure
from PIL import Image
from natsort import natsorted
from filecmp import dircmp
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from lib import dataset, network, encoder, recurrent_module, decoder, loss, vis, utils


# inspired by numpy move axis function
# def tf_move_axis(X, src, dst):
#     ndim = len(X.get_shape())
#     order = [for i in range(ndim)]


def to_npy(out_dir, arr): # 用于将数据以npy的形式存储
    np.save(out_dir, arr) # 将arr数组以二进制形式存储在out_dir中(会自动加上.npy)


def load_npy(npy_path):
    if isinstance(npy_path, str): # isinstance()的功能是判断npy_path是否为str类型
        return np.expand_dims(np.load(npy_path), 0) # numpy.load()的功能是载入npy_path路径的npy数据；numpy.expand_dims()的功能是在第axis维将数据加上去
    ret = []
    for p in npy_path:
        ret.append(np.load(p))
    return np.stack(ret)


def is_epoch_dir(epoch_dir):
    return "epoch_" in epoch_dir


def get_latest_epoch_index(model_dir): # 用于找到最近的一个epoch的索引
    if is_epoch_dir(model_dir):
        model_dir = os.path.dirname(model_dir)
    i = 0
    while os.path.exists(os.path.join(model_dir, "epoch_{}".format(i))):
        i += 1
    return i-1


def get_latest_epoch(model_dir):
    return model_dir + "\\epoch_{}".format(get_latest_epoch_index(model_dir))


def get_latest_loss(model_dir, loss_type="train"):
    epoch = get_latest_epoch(model_dir)
    epoch_prev = model_dir + \
        "\\epoch_{}".format(get_latest_epoch_index(model_dir)-1)

    try:
        return np.load(epoch+"\\{}_loss.npy".format(loss_type))
    except:
        return np.load(epoch_prev+"\\{}_loss.npy".format(loss_type))


def get_model_params(model_dir):
    json_list = dataset.construct_file_path_list_from_dir(model_dir, ".json")
    if json_list:
        return read_params(json_list[0])
    return {}


def get_model_predictions(obj_id, model_dir):
    epoch_count = get_latest_epoch_index(model_dir)+1 # 最近的一个epoch是第几个
    x, y = dataset.load_obj_id(grep_obj_id(obj_id)) # grep_obj_id得到
    for i in range(epoch_count): # 循环epoch次
        net = network.Network_restored("{}\\epoch_{}".format(model_dir, i)) # 载入网络参数
        yp = net.predict(x)


def get_model_dataset_split(model_dir):
    try:
        X_train = np.load("{}\\X_train.npy".format(model_dir))
    except:
        X_train = None

    try:
        X_train = np.load("{}\\X_train.npy".format(model_dir))
    except:
        X_train = None

    try:
        y_train = np.load("{}\\y_train.npy".format(model_dir))
    except:
        y_train = None

    try:
        X_val = np.load("{}\\X_val.npy".format(model_dir))
    except:
        X_val = None

    try:
        y_val = np.load("{}\\y_val.npy".format(model_dir))
    except:
        y_val = None

    try:
        X_test = np.load("{}\\X_test.npy".format(model_dir))
    except:
        X_test = None
    try:
        y_test = np.load("{}\\y_test.npy".format(model_dir))
    except:
        y_test = None

    return X_train, X_val, X_test, y_train, y_val, y_test


def filter_files(regex):
    return natsorted(glob.glob(regex, recursive=True))


def list_folders(path="."):
    folder_list = sorted(next(os.walk(path))[1])
    ret = []
    for f in folder_list:
        ret.append(path+"\\"+f)
    return ret


def check_params_json(param_json="params.json"): # 用于检查params_json文件是否存在，若是不存在，就写一个空的
    # os.path.exists()方法用于判断文件/文件夹是否存在
    if not os.path.exists(param_json): # 如果param_json不存在
        param_data = {} # 定义一个字典
        with open(param_json, 'w') as param_file:
            # json.dumps的功能是将dict转化为str格式，json.loads的功能是将str转化为dict格式
            # json.dump和json.load也是类似的功能，只是与文件操作结合起来了
            json.dump(param_data, param_file) # 将param_data转化为str写到param_file中去


def read_params(params_json="params.json"): # 用作读取超参，默认存储超参的文件名为"params.json"
    check_params_json(params_json) # 检查param_json文件是否存在，若是不存在会写一个空的
    # open()的功能是打开一个文件，创建一个file对象，相关的方法才可以调用它进行读写
    # file.read()的功能是从文件读取指定的字节数，如果没有给定或者为负则读取所有内容
    # json.load()的功能是从文件中读取str并转化为dict
    return json.loads(open(params_json).read())


def fix_nparray(path):
    arr = np.load(path)
    N = len(arr)
    l = arr[0]
    for i in range(1, N):
        l += arr[i]
    np.save(path, np.array(l))


def replace_with_flat(path):
    arr = np.load(path)
    np.save(path, arr.flatten())


def grep_params(s):
    regex = "^.*=(.*)$"
    return re.findall(regex, s)[0]


def grep_epoch_name(epoch_dir): # 用于再epoch_dir这个字符串中取出epoch_*
    return re.search(".*(epoch_.*).*", epoch_dir).group(1)


def grep_learning_rate(s):
    regex = "^.*L:(.*?)_.*$"
    return float(re.findall(regex, s)[0])


def grep_batch_size(s):
    regex = "^.*B:(.*?)_.*$"
    return float(re.findall(regex, s)[0])


def grep_epoch_count(s):
    regex = "^.*E:(.*?)_.*$"
    return float(re.findall(regex, s)[0])


def grep_obj_id(s): # 用于得到 类别_模型名 字符串
    s = os.path.basename(s) # 返回path最后的文件名。如果path以／或\结尾，那么就会返回空值
    regex = "(.*_.*)_(x|y|yp|p|sm).(png|npy)$" # 正则表达式
    return re.search(regex, s).group(1) # 得到第一个括号里的字符串，即 类别_模型名，如 02691156_1a04e3eab45ca15dd86060f189eb133


def grep_stepcount(s):
    s = os.path.basename(s)
    regex = "(.*)_(.*_.*)_(x|y|yp|p|sm).(png|npy)$"
    return re.search(regex, s).group(1)


def grep_timestamp(s):
    regex = ".*model_(.*)_(.*)"
    ret = re.search(regex, s)
    return ret.group(1), ret.group(2)


def make_dir(file_dir): # 用于创建路径
    # os.path.isdir()判断参数是否是路径，需要注意的是，其中的参数必须是绝对路径
    if not os.path.isdir(file_dir): # 如果该路径不是路径，即不存在该路径
        os.makedirs(file_dir) # 创建路径


def make_prev_dirs(file_dir):
    file_dir = os.path.dirname(file_dir)
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)


def clean_dir(file_dir):
    if os.path.isdir(file_dir):
        shutil.rmtree(file_dir)
        os.makedirs(file_dir)


def get_file_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def hstack(a, b):
    return np.hstack((a, b))


def vstack(a, b):
    return np.vstack((a, b))


def get_summary_as_array(model_dir, run="train", scalar="loss"):
    name = "\\{}_{}.npy".format(run, scalar)
    if os.path.exists(model_dir+name):
        return np.load(model_dir+name)

    event_file_path = glob.glob(model_dir+"\\{}\\event*".format(run))[0]
    event_acc = EventAccumulator(event_file_path)
    event_acc.Reload()
    ret = [[s.step, s.value] for s in event_acc.Scalars(scalar)]
    # np.save(model_dir+name, ret)

    return ret
