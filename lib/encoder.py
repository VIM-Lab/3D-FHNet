import tensorflow as tf
import numpy as np
from lib import utils
from numpy.random import choice
from lib.basic_layers import ResidualBlock
from lib.attention_module import AttentionModule


# 用于卷积sequence
def conv_sequence(sequence, in_featuremap_count, out_featuremap_count, initializer=None, K=3, S=[1, 1, 1, 1], D=[1, 1, 1, 1], P="SAME"):
    with tf.name_scope("conv_sequence"): # 在conv_sequence名字域中
        if initializer is None: # 如果没有给定初始化器
            init = tf.contrib.layers.xavier_initializer() # 就使用Xavier初始化器
        else: # 反之，如果给定了初始化器
            init = initializer # 使用给定的初始化器

        kernel = tf.Variable(
            init([K, K, in_featuremap_count, out_featuremap_count]), name="kernel") # 创建卷积核的变量
        bias = tf.Variable(init([out_featuremap_count]), name="bias") # 创建偏差的变量

        # tf.nn.conv2d(input,filter,strides,padding)中，
        # input是一个4D输入[batch_size,in_height,in_width,n_channels],
        # filter是一个4D输入[filter_height,filter_width,in_channels,out_channels]
        # padding有两种选项，same是做填充，valid是不做填充
        # tf.nn.bias_add()的功能是将bias加到每一个value上
        def conv2d(x): return tf.nn.bias_add(tf.nn.conv2d(
            x, kernel, S, padding=P, dilations=D, name="conv2d"), bias)
        # tf.map_fn(fn,elems)用于将elems从第一维展开，批量进行fn运算
        ret = tf.map_fn(conv2d, sequence, name="conv2d_map")
        # tf.add_to_collection('list_name',element)用于将element添加到list_name中
        # tf.get_collection('list_name')返回名称为list_name的列表
        tf.add_to_collection("feature_maps", ret)

        # visualization code
        params = utils.read_params() # 读取超参
        image_count = params["VIS"]["IMAGE_COUNT"]
        if params["VIS"]["KERNELS"]:
            kern_1 = tf.concat(tf.unstack(kernel, axis=-1), axis=-1)
            kern_2 = tf.transpose(kern_1, [2, 0, 1])
            kern_3 = tf.expand_dims(kern_2, -1)
            tf.summary.image("2d kernel", kern_3, max_outputs=image_count)

        if params["VIS"]["FEATURE_MAPS"]:
            feature_map_1 = tf.concat(tf.unstack(ret, axis=4), axis=2)
            feature_map_2 = tf.concat(
                tf.unstack(feature_map_1, axis=1), axis=2)
            feature_map_3 = tf.expand_dims(feature_map_2, -1)
            tf.summary.image("feature_map", feature_map_3,
                             max_outputs=image_count)

        if params["VIS"]["HISTOGRAMS"]:
            tf.summary.histogram("kernel", kernel)
            tf.summary.histogram("bias", bias)

        if params["VIS"]["SHAPES"]:
            print(ret.shape)
    return ret

# 用于对sequence进行attention_residual操作
def attention_sequence(sequence, in_featuremap_count, out_featuremap_count):
    with tf.name_scope("attention_sequence"): # 在conv_sequence名字域中

        attention_module = AttentionModule()
        residual_block = ResidualBlock()

        def attention_def(x):
            x = attention_module.f_prop(x, input_channels=in_featuremap_count)
            return x
        def residual_def(x):
            x = residual_block.f_prop(x, input_channels=in_featuremap_count, output_channels=out_featuremap_count)
            return x
        
        # tf.map_fn(fn,elems)用于将elems从第一维展开，批量进行fn运算
        ret = tf.map_fn(attention_def, sequence, name="attention_module_map")
        ret = tf.map_fn(residual_def, sequence, name="residual_block_map")
        
        # tf.add_to_collection('list_name',element)用于将element添加到list_name中
        # tf.get_collection('list_name')返回名称为list_name的列表
        tf.add_to_collection("feature_maps", ret)

    return ret

def max_pool_sequence(sequence, K=[1, 2, 2, 1], S=[1, 2, 2, 1], P="VALID"):
    with tf.name_scope("max_pool_sequence"): # 在max_pool_sequence名字域内
        def max_pool(a): return tf.nn.max_pool(a, K, S, padding=P) # 定义一个max_pool函数
        ret = tf.map_fn(max_pool, sequence, name="max_pool_map") # 将max_pool函数批量应用于sequence
    return ret


def relu_sequence(sequence):
    with tf.name_scope("relu_sequence"): # 在relu_sequence名字域中
        # tf.map_fn(fn,elems)用于将elems从第一维展开，批量进行fn运算
        ret = tf.map_fn(tf.nn.relu, sequence, name="relu_map") # 将relu批量应用于sequence
    return ret


def fully_connected_sequence(sequence, initializer=None):
    with tf.name_scope("fully_connected_sequence"): # 在fully_connected_sequence名字域内
        if initializer is None: # 如果没有给定初始化器
            init = tf.contrib.layers.xavier_initializer() # 就用Xavier初始化器
        else: # 反之，如果给定了初始化器
            init = initializer # 就使用给定的初始化器

        weights = tf.Variable(
            init([1024, 1024]), name="weights") # 定义weights变量
        bias = tf.Variable(init([1024]), name="bias") # 定义bias变量

        def forward_pass(a): return tf.nn.bias_add( # 定义前馈函数，将输入与weights相乘再加上bias
            tf.matmul(a, weights), bias)

        ret = tf.map_fn(forward_pass, sequence, name='fully_connected_map') # 将前馈操作批量应用于sequence

        params = utils.read_params()
        if params["VIS"]["HISTOGRAMS"]:
            tf.summary.histogram("weights", weights)
            tf.summary.histogram("bias", bias)

    return ret


def flatten_sequence(sequence):
    with tf.name_scope("flatten_sequence"): # 在flatten_sequence名字域内
        ret = tf.map_fn( # tf.contrib.layers.flatten()用于将sequence展开成一维向量，但是会保留batch_size维
            tf.contrib.layers.flatten,  sequence, name="flatten_map")
    return ret


# sequence是输入，in_featuremap_count是输入的特征图个数（厚度，层数），out_featuremap_count是输出的特征图个数，k是卷积核大小，d是
def block_residual_encoder(sequence, in_featuremap_count, out_featuremap_count,  K_1=3, K_2=3, K_3=1, D=[1, 1, 1, 1], initializer=None, pool=True):
    with tf.name_scope("block_residual_encoder"): # 在block_residual_encoder名字域内
        if initializer is None: # 如果没有给定初始化器
            init = tf.contrib.layers.xavier_initializer() # 就使用Xavier初始化器
        else: # 反之，如果给定了初始化器
            init = initializer # 就使用给定的初始化器

        if K_1 != 0:
            conv1 = conv_sequence(sequence, in_featuremap_count, out_featuremap_count, K=K_1, D=D, initializer=init) # 第一个3*3卷积
            relu1 = relu_sequence(conv1) # 通过relu激活函数
            out = relu1

        if K_2 != 0:
            conv2 = conv_sequence(out, out_featuremap_count, out_featuremap_count, K=K_2, D=D, initializer=init) # 第二个3*3卷积
            relu2 = relu_sequence(conv2) # 通过relu激活函数
            out = relu2

        if K_3 != 0:    
            conv3 = conv_sequence(sequence, in_featuremap_count, out_featuremap_count, K=K_3, D=D, initializer=init) # 1*1卷积匹配大小
            out = conv3 + out # 残差连接

        if pool: # 如果需要池化
            pool = max_pool_sequence(out) # 进行池化
            out = pool # 赋值给out
   
        return out





def block_residual_attention_encoder(sequence, in_featuremap_count, out_featuremap_count):
    with tf.name_scope("block_residual_attention_encoder"): # 在block_residual_encoder名字域内
        # if initializer is None: # 如果没有给定初始化器
        #     init = tf.contrib.layers.xavier_initializer() # 就使用Xavier初始化器
        # else: # 反之，如果给定了初始化器
        #     init = initializer # 就使用给定的初始化器

        out = attention_sequence(sequence, in_featuremap_count, out_featuremap_count)

        out = max_pool_sequence(sequence, K = [1, 3, 3, 1], S = [1, 2, 2, 1], P = 'SAME')

        return out


class Residual_Encoder:
    def __init__(self, sequence, feature_map_count=[96, 128, 256, 256, 256, 256], initializer=None):
        with tf.name_scope("Residual_Encoder"): # 在Residual_Encoder名字域下
            if initializer is None: # 如果没有给定初始化器
                init = tf.contrib.layers.xavier_initializer() # 就使用xavier初始化器
            else: # 反之，如果给定了初始化器
                init = initializer # 就使用给定的

            cur_tensor = block_residual_encoder( # 第一个残差卷积块
                sequence, 3, feature_map_count[0], K_1=7, K_2=3, K_3=0, initializer=init)
            # convolution stack
            N = len(feature_map_count) # N取feature_map_count的个数
            for i in range(1, N): # 循环N-1次
                cur_tensor = block_residual_encoder( # 第2-N个残差卷积块
                    cur_tensor, feature_map_count[i-1], feature_map_count[i], initializer=init)

            self.out_tensor = cur_tensor


class Residual_Attention_Encoder:
    def __init__(self,sequence, feature_map_count=[96, 128, 256, 256, 256, 256], initializer=None):
        with tf.name_scope("Residual_Attention_Encoder"): # 在Residual_Attention_Encoder名字域下
            if initializer is None: # 如果没有给定初始化器
                init = tf.contrib.layers.xavier_initializer() # 就使用xavier初始化器
            else: # 反之，如果给定了初始化器
                init = initializer # 就使用给定的

            cur_tensor = block_residual_encoder( # 第一个残差卷积块128->64
                sequence, 3, feature_map_count[0], K_1=7, K_2=3, K_3=0, initializer=init)

            cur_tensor = block_residual_encoder( # 第二个残差卷积块64->32
                cur_tensor, feature_map_count[0], feature_map_count[1], K_1=3, K_2=3, K_3=1, initializer=init)

            cur_tensor = block_residual_encoder( # 第三个残差卷积块32->16
                cur_tensor, feature_map_count[1], feature_map_count[2], K_1=3, K_2=3, K_3=1, initializer=init)


            N = len(feature_map_count) # N取feature_map_count的个数
            for i in range(3, N): # 循环N-1次
                cur_tensor = block_residual_attention_encoder( # 第2-N个残差卷积块
                    cur_tensor, feature_map_count[i-1], feature_map_count[i])

            self.out_tensor = cur_tensor

