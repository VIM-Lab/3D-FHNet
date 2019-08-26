import tensorflow as tf
from lib import utils


def shuffle_sequence(value):
    with tf.name_scope("shuffle_sequence"):
        ret = tf.transpose(value, [1, 0, 2, 3, 4])
        ret = tf.random_shuffle(ret)
        ret = tf.transpose(ret, [1, 0, 2, 3, 4])
    return value


class Preprocessor():
    def __init__(self, X):
        with tf.name_scope("Preprocessor"): # 使用Preprocessor名字域
            params = utils.read_params() # 读取超参
            if params["TRAIN"]["TIME_STEP_COUNT"] == "RANDOM": # 如果超参中训练时的time_step_count是RANDOM
                n_timesteps = tf.random_uniform( # tf.random_uniform()用于产生随机数，并且产生的值是均匀分布在minval和maxval之间
                    [], minval=1, maxval=13, dtype=tf.int32) # 如果要随机次数训练，那么在minval和maxval之间随机取 TODO: shape属性为[]会出现什么情况？
                tf.summary.scalar("n_timesteps", n_timesteps) # 可视化
            elif isinstance(params["TRAIN"]["TIME_STEP_COUNT"], int) and params["TRAIN"]["TIME_STEP_COUNT"] > 0: # 如果超参中的time_step_count不是RANDOM而是一个int数
                n_timesteps = params["TRAIN"]["TIME_STEP_COUNT"] # 使用设置好的那个数来作为n_timesteps
            else: # 又不是随机又不是特定整数
                n_timesteps = tf.shape(X)[1] # n_timesteps取X的第一维 TODO： 估计是取所有图片

            n_batchsize = tf.shape(X)[0] # n_batchsize取X的第0维
            X_dropped_alpha = X[:, :, :, :, 0:3]  # 取RGB数据，丢弃透明度
            X_cropped = tf.random_crop( # 裁剪X
                X_dropped_alpha, [n_batchsize, n_timesteps, 128, 128, 3])   # randomly crop

            if params["TRAIN"]["SHUFFLE_IMAGE_SEQUENCE"]: # 如果超参里设置了打乱图片的顺序
                X_shuffled = shuffle_sequence(X_cropped) # 打乱
                self.out_tensor = X_shuffled # 将待返回的tensor设置成打乱的X
            else: # 如果超参里设置的是不打乱图片的顺序
                self.out_tensor = X_cropped # 将待返回的tensor设置成裁剪后的X
