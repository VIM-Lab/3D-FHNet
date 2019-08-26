import tensorflow as tf
from lib import utils


def conv_vox(vox, in_featurevoxel_count, out_featurevoxel_count, K=3, S=[1, 1, 1, 1, 1], D=[1, 1, 1, 1, 1], initializer=None, P="SAME"):
    with tf.name_scope("conv_vox"): # 在conv_vox名字域内
        if initializer is None: # 如果没有给定初始化器
            init = tf.contrib.layers.xavier_initializer() # 就使用Xavier初始化器
        else: # 反之如果给定了初始化器
            init = initializer # 就使用给定的初始化器

        kernel = tf.Variable( # 定义卷积核大小
            init([K, K, K, in_featurevoxel_count, out_featurevoxel_count]), name="kernel")
        bias = tf.Variable(init([out_featurevoxel_count]), name="bias") # 定义偏差

        def conv3d(x): return tf.nn.bias_add(tf.nn.conv3d( # 卷积运算
            x, kernel, S, padding=P, dilations=D, name="conv3d"), bias)
        # tf.map_fn(fn,elems)用于将elems从第一维展开，批量进行fn运算
        ret = tf.map_fn(conv3d, vox, name="conv2d_map")

        tf.add_to_collection("feature_voxels", ret) # 储存

        # visualization code
        params = utils.read_params()
        image_count = params["VIS"]["IMAGE_COUNT"]
        if params["VIS"]["KERNELS"]:
            kern_1 = tf.concat(tf.unstack(kernel, axis=-1), axis=-1)
            kern_2 = tf.transpose(kern_1, [3, 0, 1, 2])
            kern_3 = tf.expand_dims(kern_2, -1)
            kern_4 = tf.concat(tf.unstack(kern_3, axis=1), axis=1)
            tf.summary.image("3d kernel", kern_4, max_outputs=image_count)

        if params["VIS"]["VOXEL_SLICES"]:
            vox_slice_1 = tf.unstack(ret, axis=4)[1]
            vox_slice_2 = tf.split(vox_slice_1, 4, axis=3)
            vox_slice_3 = tf.concat(vox_slice_2, axis=1)
            vox_slice_4 = tf.concat(tf.unstack(vox_slice_3, axis=-1), axis=2)
            vox_slice_5 = tf.expand_dims(vox_slice_4, -1)
            tf.summary.image("vox_slices", vox_slice_5,
                             max_outputs=image_count)

        if params["VIS"]["FEATURE_VOXELS"]:
            tf.summary.tensor_summary("feature_voxels", ret[0, :, :, :, 0])

        if params["VIS"]["HISTOGRAMS"]:
            tf.summary.histogram("kernel", kernel)
            tf.summary.histogram("bias", bias)

        if params["VIS"]["SHAPES"]:
            print(ret.shape)

    return ret


def unpool_vox(value):  # from tenorflow github board
    with tf.name_scope('unpool_vox'):
        def unpool_vox(x):
            sh = x.get_shape().as_list() # 获取value的维度并转换成list
            dim = len(sh[1: -1]) # 取1-倒数第二个为维度
            out = (tf.reshape(x, [-1] + sh[-dim:])) # [-dim:]切片得到后面3个维度，在前面连上-1，即自动计算第一个位置需要的维度

            for i in range(dim, 0, -1): # 从dim到0循环，左闭右开，第三个数是步长
                out = tf.concat([out, tf.zeros_like(out)], i) # tf.zeros_like()得到形状一样的值全为0的tensor

            out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]] # 得到想要的池化后尺寸，即n_cell都翻倍
            out = tf.reshape(out, out_size) # reshape得到最终池化后的结果
            return out

        ret = tf.map_fn(unpool_vox, value, name="max_pool_map")

    return ret


def relu_vox(vox):
    with tf.name_scope("relu_vox"):
        ret = tf.map_fn(tf.nn.relu, vox, name="relu_map")
    return ret


def block_simple_decoder(vox, in_featurevoxel_count, out_featurevoxel_count, K=3, D=[1, 1, 1, 1, 1], initializer=None, unpool=False):
    with tf.name_scope("block_simple_decoder"):
        if initializer is None:
            init = tf.contrib.layers.xavier_initializer()
        else:
            init = initializer

        conv = conv_vox(vox, in_featurevoxel_count, out_featurevoxel_count,
                        K=K,  D=D, initializer=init)
        if unpool:
            out = relu_vox(unpool_vox(conv))
        else:
            out = relu_vox(conv)

    return out


def block_residual_decoder(vox, in_featurevoxel_count, out_featurevoxel_count, K_1=3, K_2=3, K_3=1, D=[1, 1, 1, 1, 1], initializer=None, unpool=False):
    with tf.name_scope("block_residual_decoder"): # 在block_residual_decoder名字域内
        if initializer is None: # 如果没有指定初始化器
            init = tf.contrib.layers.xavier_initializer() # 就使用Xavier初始化器
        else: # 反之如果指定了初始化器
            init = initializer # 就使用指定的初始化器

        if K_1 != 0:
            conv1 = conv_vox(vox, in_featurevoxel_count, out_featurevoxel_count, K=K_1, D=D, initializer=init) # 第一个3*3*3卷积
            relu1 = relu_vox(conv1) # relu激活函数
            out = relu1

        if K_2 != 0:
            conv2 = conv_vox(out, out_featurevoxel_count, out_featurevoxel_count, K=K_2, D=D, initializer=init) # 第二个3*3*3卷积
            relu2 = relu_vox(conv2) # relu激活函数
            out = relu2

        if K_3 != 0:
            conv3 = conv_vox(vox, in_featurevoxel_count, out_featurevoxel_count, K=K_3, D=D, initializer=init) #1*1*1卷积匹配大小
            out = conv3 + out # 残差连接

        if unpool:
            unpool = unpool_vox(out)
            out = unpool

    return out


class Residual_Decoder: # 残差解码器
    def __init__(self, hidden_state, feature_vox_count=[128, 128, 128, 64, 32, 2], initializer=None):
        with tf.name_scope("Residual_Decoder"): # 在Residual_Decoder名字域内
            if initializer is None: # 如果没有给定初始化器
                init = tf.contrib.layers.xavier_initializer() # 就使用Xavier初始化器
            else: # 反之如果给定了初始化器
                init = initializer # 就使用给定的初始化器

            N = len(feature_vox_count) # decoder的层数
            hidden_shape = hidden_state.get_shape().as_list() # 获取hidden_state的维度，并转换成list
            cur_tensor = unpool_vox(hidden_state) # 反池化体素，边长变成原来的2倍（多出来的格子都是0）
            cur_tensor = block_residual_decoder(
                cur_tensor, hidden_shape[-1], feature_vox_count[0], initializer=init)
            for i in range(1, N-1):
                unpool = True if i <= 2 else False
                cur_tensor = block_residual_decoder(
                    cur_tensor, feature_vox_count[i-1], feature_vox_count[i], initializer=init, unpool=unpool)

            self.out_tensor = conv_vox(
                cur_tensor, feature_vox_count[-2], feature_vox_count[-1], initializer=init)


class Dilated_Decoder:
    def __init__(self, hidden_state, feature_vox_count=[128, 128, 128, 64, 32, 2], initializer=None):
        with tf.name_scope("Dilated_Decoder"):
            if initializer is None:
                init = tf.contrib.layers.xavier_initializer()
            else:
                init = initializer

            N = len(feature_vox_count)
            hidden_shape = hidden_state.get_shape().as_list()
            cur_tensor = unpool_vox(hidden_state)
            cur_tensor = block_simple_decoder(
                cur_tensor, hidden_shape[-1], feature_vox_count[0], initializer=init)
            for i in range(1, N-1):
                unpool = True if i <= 2 else False
                cur_tensor = block_simple_decoder(
                    cur_tensor, feature_vox_count[i-1], feature_vox_count[i], D=[1, 2, 2, 2, 1], initializer=init, unpool=unpool)

            self.out_tensor = conv_vox(
                cur_tensor, feature_vox_count[-2], feature_vox_count[-1], initializer=init)


class Simple_Decoder:
    def __init__(self, hidden_state, feature_vox_count=[128, 128, 128, 64, 32, 2], initializer=None):
        with tf.name_scope("Simple_Decoder"):
            if initializer is None:
                init = tf.contrib.layers.xavier_initializer()
            else:
                init = initializer

            N = len(feature_vox_count)
            hidden_shape = hidden_state.get_shape().as_list()
            cur_tensor = unpool_vox(hidden_state)
            cur_tensor = block_simple_decoder(
                cur_tensor, hidden_shape[-1], feature_vox_count[0], initializer=init)
            for i in range(1, N-1):
                unpool = True if i <= 2 else False
                cur_tensor = block_simple_decoder(
                    cur_tensor, feature_vox_count[i-1], feature_vox_count[i], initializer=init, unpool=unpool)

            self.out_tensor = conv_vox(
                cur_tensor, feature_vox_count[-2], feature_vox_count[-1], initializer=init)
