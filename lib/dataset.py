"""deals with data for project"""
import re
import json
import os
import sys
import math
import random
import tarfile
import numpy as np
import pandas as pd
import trimesh
import tensorflow as tf
from PIL import Image
from filecmp import dircmp
from collections import deque
from third_party import binvox_rw
from lib import utils, dataset
from sklearn import model_selection
from keras.utils import to_categorical
from numpy.random import randint, permutation, shuffle
from natsort import natsorted


image_dir='ShapeNetRendering'

def load_obj_id(obj_id):
    data_path, label_path = id_to_path(obj_id)
    return load_imgs_from_dir(data_path), np.squeeze(load_voxs_from_dir(label_path))


def id_to_path(obj_id, data_dir=".\\data\\{}\\".format(image_dir), label_dir=".\\data\\ShapeNetVox32\\"):
    regex = re.search("(.*)_(.*)", obj_id)
    ret_1 = os.path.join(data_dir, regex.group(1), regex.group(2))
    ret_2 = os.path.join(label_dir, regex.group(1), regex.group(2))
    return ret_1, ret_2


# loading functions
def load_img(img_path): # 用于加载单张图片，参数是图片的路径，返回以np.ndarray存储的图片
    # np.array()用于创建np.ndarray数组
    im = Image.open(img_path)
    # im = tf.image.resize_image_with_crop_or_pad(im, 128, 128)
    return np.array(im) # 加载图片，返回一个np.ndarray


def load_vox(vox_path): # 用于加载单个体素，参数是体素的路径，返回以np.ndarray存储的体素
    with open(vox_path, 'rb') as f: # rb是以二进制读形式打开
        # third_party.binvox_rw.read_as_3d_array().data将体素数据读取成三维数组，具体进入.\third_party\binvox_rw.py中查看
        # 其中，三维数组代表体素的x，y，z。占用的体素值为true，非占用的体素值为false
        # keras.utils.to_categorical()用于将数据转换成0，1独热编码的形式
        # 在这个例子中，to_categorical将false转换成[1,0],将true转换成[0,1]
        return to_categorical(binvox_rw.read_as_3d_array(f).data) # 返回转换成独热编码的数组


def load_imgs(img_path_list): # 用于加载一批图片，参数是图片的路径的list，返回二维np.ndarray存储的一批图片
    # assert()用于检查括号中的条件是否为true，如果为false，就会raise一个Error
    assert(isinstance(img_path_list, (list, np.ndarray))) # 检查传进函数的参数img_path_list是否是一个list或者np.ndarray

    ret = [] # 创建一个list
    for p in img_path_list: # 遍历img_path_list
        ret.append(load_img(p)) # 将加载的图片添加到ret列表中
    return np.stack(ret) # np.stack()是用于合并list或者改变list的索引，但是此处的功能应该只是将其变成np.ndarray


def load_voxs(vox_path_list): # 用于加载一批体素，参数是体素的路径list，返回一个二维np.ndarray
    assert(isinstance(vox_path_list, (list, np.ndarray))) # 检查传进函数的参数vox_path_list是否是一个list或者np.ndarray

    ret = [] # 创建一个list
    for p in vox_path_list: # 遍历vox_path_list
        ret.append(load_vox(p)) # 将加载的体素添加到ret列表中
    return np.stack(ret) # np.stack()是用于合并list或者改变list的索引，但是此处的功能应该只是将其变成np.ndarray


def load_imgs_from_dir(img_dir):
    img_path_list = construct_file_path_list_from_dir(img_dir, [".png"])
    return load_imgs(img_path_list)


def load_voxs_from_dir(vox_dir):
    vox_path_list = construct_file_path_list_from_dir(vox_dir, [".binvox"])
    return load_voxs(vox_path_list)


# #  dataset loading functions
def load_data(data_samples): # 用于加载图片
    if isinstance(data_samples, str): # 如果传进函数的data_samples是一个string实例
        data_samples = [data_samples] # 将其转换成只包含它自己的list
    return load_imgs(data_samples) # 返回加载的图片


def load_label(label_samples): # 用于加载体素
    if isinstance(label_samples, str): # 如果传入的参数是一个string
        label_samples = [label_samples] # 就将其转换成只包含它一个元素的list
    # numpy.squeeze()可以用于删除数组中的只有一个元素的维度
    return np.squeeze(load_voxs(label_samples))


# load preprocessed data and labels
def load_preprocessed_dataset(): # 用于取得所有打包的数据的路径
    data_preprocessed_dir = utils.read_params( # 读取预处理过的数据的路径
    )["DIRS"]["DATA_PREPROCESSED"]

    data_all = sorted( # sorted()函数对所有可迭代的对象进行排序操作
        dataset.construct_file_path_list_from_dir(data_preprocessed_dir, ["_x.npy"])) # 寻找data_preprocessed_dir目录下的所有文件名包括'_x.npy'的文件名
    label_all = sorted( # sorted()函数对所有可迭代的对象进行排序操作
        dataset.construct_file_path_list_from_dir(data_preprocessed_dir, ["_y.npy"])) # 寻找data_preprocessed_dir目录下的所有文件名包括'_y.npy'的文件名

    return np.array(data_all), np.array(label_all) # 将list转换成np.ndarray并返回


def load_random_sample():
    data, label = load_preprocessed_dataset()
    i = randint(0, len(data))
    return np.load(data[i]), np.load(label[i])


def load_testset(model_dir):
    try:
        X_test = np.load(
            "{}\\X_test.npy".format(model_dir))
        y_test = np.load(
            "{}\\y_test.npy".format(model_dir))
    except:
        model_dir = os.path.dirname(model_dir)
        X_test = np.load(
            "{}\\X_test.npy".format(model_dir))
        y_test = np.load(
            "{}\\y_test.npy".format(model_dir))

    return X_test, y_test


def shuffle_batchs(data, label, batch_size):
    # print(data, label, batch_size)
    assert(len(data) == len(label)) # 如果图片和体素的路径个数对不上，就报错
    num_of_batches = math.ceil(len(data)/batch_size) # 计算batch的个数
    perm = permutation(len(data)) # 随机排序
    data_batchs = np.array_split(data[perm], num_of_batches) # data[perm]得到的是按perm序号排列的乱序的data，np.array_split()将数据分成num_of_batches份
    label_batchs = np.array_split(label[perm], num_of_batches) # 对label做与data一样的操作

    return deque(data_batchs), deque(label_batchs) # 将分成批的data和label转换成deque返回


def train_val_test_split(data, label, split=0.1):
    # split into training and test set
    X_train, X_test, y_train, y_test = model_selection.train_test_split( # 划分图片和体素数据的路径，以9：1划分成训练集和测试集
        data, label, test_size=split)  # shuffled
    # split of validation set
    X_train, X_val, y_train, y_val = model_selection.train_test_split( # 将训练集再按9：1划分成训练集和验证集
        X_train, y_train, test_size=split)  # shuffled

    # 得到的训练集、验证集、测试集的比例为0.81：0.09：0.10
    return X_train, y_train, X_val, y_val, X_test, y_test


def setup_dir(): # 用于创建一些需要的路径
    params = utils.read_params() # 读取超参
    DIR = params["DIRS"] # 从读取的超参中读取DIR字典
    for d in DIR.values(): # 遍历DIR字典
        utils.make_dir(d) # 创建路径

    utils.check_params_json("params.json") # 用于检查params_json文件是否存在，若是不存在，就写一个空的

# 用于寻找dir路径下的文件中，含有file_filter中名字的子串的文件的路径的list或tuple。
# 如，寻找dir目录下的所有.png文件的路径的list（file_filter='.png'）
def construct_file_path_list_from_dir(dir, file_filter): 

    if isinstance(file_filter, str): # 如果file_filter是一个字符串类型
        file_filter = [file_filter] # 把字符串变成一个含有单个字符串的list
    paths = [[] for _ in range(len(file_filter))] # 创建一个file_filter中元素个数长度的list的list，即二维list，分别存储每个file_filter后缀的文件的路径

    # os.walk()用于遍历一个目录，参数是需要遍历的路径，返回一个三元组，
    # 分别是root（正在遍历的这个文件夹的地址），dirs（一个list，内容是root包含的所有目录的名字），files（一个list，内容是root包含的所有文件的名字）
    for root, _, files in os.walk(dir): # 遍历dir
        for f_name in files: # 遍历dir路径下的所有文件的名字
            for i, f_substr in enumerate(file_filter): # 枚举file_filter
                if f_substr in f_name: # 如果file_filter中的字符串是file中某个文件名的子串
                    (paths[i]).append(root + '\\' + f_name) # 将这个文件的路径加入到paths中去

    for i, p in enumerate(paths): # 枚举paths
        # natsorted()用于将字符串序列自然排序，如'a1'在'a02'前面，'a3'在'a12'前面
        paths[i] = natsorted(p) # 将paths中的每一行都按自然顺序排序

    if len(file_filter) == 1: # 如果file_filter中只有一个元素
        return paths[0] # 返回路径的list

    return tuple(paths) # 如果file_filter中有多个元素，返回路径的list的元组


def create_path_csv(data_dir, label_dir):
    print("creating path csv for {} and {}".format(data_dir, label_dir)) # 在终端打印创建csv的信息
    params = utils.read_params() # 读取超参

    common_paths = [] # 创建一个list，用于存储data和label中相同的对应的路径
    # dircmp用于比较两个目录中的文件是否相同,得到filecmp.dircmp object，.common_dirs可以得到两个目录中的相同目录
    # dircmp().subdirs返回相同名称的对应字典，键是相同文件夹的名字，值是filecmp.dircmp object，.common_dirs可以得到那个相同目录下的相同目录名，也就是两层都相同的目录的子目录名
    # dircmp().subdirs.items()得到两个目录中的相同名称的键值对
    for dir_top, subdir_cmps in dircmp(data_dir, label_dir).subdirs.items(): # 遍历data_dir和label_dir中的相同目录的键值对
        for dir_bot in subdir_cmps.common_dirs: # 遍历相同目录中的相同目录
            common_paths.append(os.path.join(dir_top, dir_bot)) # 将data_dir和label_dir中的相同目录名和相同目录中的相同目录名连起来，加到common_paths中，也就是得到两层目录都相同的list

    mapping = pd.DataFrame(common_paths, columns=["common_dirs"]) # 将common_paths列表转化成dataframe
    mapping['data_dirs'] = mapping.apply( # DataFrame.apply()用于将第一个参数的函数应用于一批数据，axis=0是按列操作，axis=1是按行操作
        lambda data_row: os.path.join(data_dir, data_row.common_dirs), axis=1) # 将data_dirs连上每个common_dirs，得到所有的对应的data的路径    
    for dirs in mapping.data_dirs:
        dirs = os.path.join(data_dir, 'rendering') # 原版需要在最后增加一个rendering文件夹

    mapping['label_dirs'] = mapping.apply( # 同上
        lambda data_row: os.path.join(label_dir, data_row.common_dirs), axis=1) # 将label_dirs连上每个common_dirs,得到所有的的对应的label的路径

    table = []
    # zip()用于将可迭代的对象打包成元组
    for n, d, l in zip(common_paths, mapping.data_dirs, mapping.label_dirs): # 将common_paths,mapping.data_dirs,mapping.label_dirs打包用于迭代
        # TODO: common_paths是由os.path.join()得到的，似乎最后没有\，所以是将路径的最后两个文件夹中间的\换成了_???
        data_row = [os.path.dirname(n)+"_"+os.path.basename(n)] # os.path.dirname()返回文件所在目录;os.path.basename()返回路径最后的文件名。这里相当于将最后一个\变成了_
        data_row += construct_file_path_list_from_dir(d, [".png"]) # data_row中增加data_dirs中所有.png文件的路径
        data_row += construct_file_path_list_from_dir(l, [".binvox"]) # data_row中增加label_dirs中所有.binvox文件的路径
        table.append(data_row) # 将这一个common_paths,data_dirs,label_dirs中得到的路径作为一行，加入到table中去

    paths = pd.DataFrame(table) # 将table转化成pandas.DataFrame,存在paths中
    paths.to_csv("{}\\paths.csv".format(params["DIRS"]["OUTPUT"])) # 将paths写成csv，存在output路径中
    return paths


def download_from_link(link):
    # os.path.basename()的功能是返回路径字符串中最后一个\之后的文件名，如果路径以\结尾，会返回空值
    # os.path.splitext()的功能是分割路径，返回路径名和文件扩展名的元组，在这里取[0]即得到不包含扩展名的文件名
    download_folder = os.path.splitext(os.path.basename(link))[0] # 下载得到的文件夹的名字是不包含扩展名的文件名
    archive = download_folder + ".tgz" # 文件的名字是文件夹名字加上.tgz扩展名

    if not os.path.isfile(archive): # os.path.isfile()用于判断路径是否是文件，此处用来判断archive文件是否存在
        os.system('wget -c {0}'.format(link)) # 如果archive文件不存在，下载link中的数据

    os.system("tar -xvzf {0}".format(archive)) # 解压archive文件
    # os.rename()的功能是将第一个参数的目录名修改成第二个参数的目录名
    os.rename(download_folder, "data\\{}".format(download_folder)) # 修改文件夹名字
    # os.system("rm -f {0}".format(archive))


def download_dataset():
    LABEL_LINK = 'ftp://cs.stanford.edu/cs/cvgl/ShapeNetVox32.tgz' # 体素数据的下载路径
    DATA_LINK = "ftp://cs.stanford.edu/cs/cvgl/ShapeNetRendering.tgz" # 图片数据的下载路径

    if not os.path.isdir("data\\ShapeNetVox32"): # 如果用于存储体素的路径不存在
        download_from_link(LABEL_LINK) # 下载体素数据

    if not os.path.isdir("data\\ShapeNetRendering"): # 如果用于存储图片的路径不存在
        download_from_link(DATA_LINK) # 下载图片数据


def preprocess_dataset():
    params = utils.read_params() # 读取超参
    dataset_size = params["DATASET_SIZE"] # 从超参中读取数据集尺寸(使用多少个模型，设置为小于等于0就是使用所有模型)
    output_dir = params["DIRS"]["OUTPUT"] # 从超参中读取输出路径
    data_preprocessed_dir = params["DIRS"]["DATA_PREPROCESSED"] # 从超参中读取处理过后的数据存放路径
    data_dir = params["DIRS"]["DATA"] # 从超参中读取数据路径

    if not os.path.isfile("{}\\paths.csv".format(output_dir)): # 判断输出路径中是否存在paths.csv
        dataset.create_path_csv( # 如果不存在，就创建一个
            "{}\\{}".format(data_dir,image_dir), "{}\\ShapeNetVox32".format(data_dir)) # paths.csv中每一行是一个物体，第一列是索引，2-倒数第2列是所有图片的路径，最后一列是体素的路径

    path_list = pd.read_csv( # 读取paths.csv
        "{}\\paths.csv".format(output_dir), index_col=0).as_matrix() # dataframe.as_matrix()用于将dataframe转换成矩阵
    # randomly pick examples from dataset
    shuffle(path_list) # numpy.random.shuffle()用于打乱list中元素的顺序

    if dataset_size <= 0 or dataset_size >= len(path_list): # 如果设定的Dataset_size小于等于零或者大于所有数据的个数
        dataset_size = len(path_list) # 就将dataset_size设定为所有数据的个数

    for i in range(dataset_size): # 循环dataset_size次
        model_name = path_list[i, 0] # 取出需要使用的路径
        utils.to_npy('{}\\{}_x'.format(data_preprocessed_dir, model_name), # 将图片数据存储成.npy
                     load_data(path_list[i, 1:-1])) # 加载图片数据
        utils.to_npy('{}\\{}_y'.format(data_preprocessed_dir, model_name), # 将体素数据存储成.npy
                     load_label(path_list[i, -1])) # 加载体素数据


def render_dataset(dataset_dir="ShapeNet", num_of_examples=None, render_count=12):
    print("[load_dataset] loading from {0}".format(dataset_dir))

    pathlist_tuple = construct_file_path_list_from_dir(
        dataset_dir, ['.obj', '.mtl'])
    pathlist = pathlist_tuple[0]  # DANGER, RANDOM
    pathlist = pathlist[:num_of_examples] if num_of_examples is not None else pathlist
    render_list = []

    for mesh_path in pathlist:
        if not os.path.isfile(mesh_path):
            continue
        try:
            mesh_obj = trimesh.load_mesh(mesh_path)
        except:
            print("failed to load {}".format(mesh_path))
            continue

        if isinstance(mesh_obj, list):
            compund_mesh = mesh_obj.pop(0)
            for m in mesh_obj:
                compund_mesh += m
        else:
            compund_mesh = mesh_obj

        render_dir = ".\\ShapeNet_Renders"
        renders = os.path.dirname(
            str.replace(mesh_path, dataset_dir, render_dir))

        if os.path.isdir(renders) and os.listdir(renders) != []:
            render_list.append(load_imgs_from_dir(renders))
        else:
            write_renders_to_disk(compund_mesh, renders, render_count)
            render_list.append(load_imgs_from_dir(renders))

    return render_list


def write_renders_to_disk(mesh, renders, render_count=10):
    print("[write_renders_to_disk] writing renders to {0} ... ".format(
        renders))
    # FIXME: stupid but clean
    os.system("rm -rf {}".format(renders))
    utils.make_dir(renders)
    scene = mesh.scene()
    for i in range(render_count):
        angle = math.radians(random.randint(15, 30))
        axis = random.choice([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        rotate = trimesh.transformations.rotation_matrix(
            angle, axis, scene.centroid)
        camera_old, _geometry = scene.graph['camera']
        camera_new = np.dot(camera_old, rotate)
        scene.graph['camera'] = camera_new
        # backfaces culled if using original trimesh package
        scene.save_image(
            '{0}\\{1}_{2}.png'.format(renders, os.path.basename(renders), i), resolution=(127, 127))

    return
