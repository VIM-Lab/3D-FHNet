import os
import sys
import re
import json
import math
import random
import keyboard
import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from lib import dataset, preprocessor, encoder, recurrent_module, decoder, loss, vis, utils
from tensorflow.python import debug as tf_debug


class Network:
    def __init__(self, params=None):
        self.read_params(params) # 读取超参
        self.place_holder() # 占位符
        self.encoder() # 编码器
        self.recurrent_module() # 循环模块
        self.decoder() # 解码器
        self.cal_loss()
        self.misc()
        self.optimizer()
        self.metric()

    def read_params(self, params=None): # 用于读取超参
        if params is None: # 如果没有给定超参
            self.params = utils.read_params() # 就读取默认超参
        else: # 如果给定了超参
            self.params = params # 就用给定的超参

        if self.params["TRAIN"]["INITIALIZER"] == "XAVIER": # 如果超参中的初始化设定为XAVIER
            self.init = tf.contrib.layers.xavier_initializer() # 用Xavier初始化（使每一层输出的方差应该尽量相等）
        else: # 如果超参中的初始化不是设定为XAVIER
            self.init = tf.random_normal_initializer() # 生成标准正态分布的随机数来初始化

        self.CREATE_TIME = datetime.now().strftime("%Y-%m-%d_%H.%M.%S") # 获取当前的时间，并将其转换为字符串格式存储下来，作为创建时间
        self.MODEL_DIR = "{}\\model_{}".format( # string.format()的功能是，将后面的参数填到前面的{}中去
            self.params["DIRS"]["MODELS_LOCAL"], self.CREATE_TIME) # 得到待创建的路径名
        utils.make_dir(self.MODEL_DIR) # 新建路径

        with open(self.MODEL_DIR + '\\params.json', 'w') as f:
            json.dump(self.params, f) # 将超参写成json文件
        
    def place_holder(self): # 占位符
        with tf.name_scope("Data"): # 定义Data的占位符
            self.X = tf.placeholder(tf.float32, [None, None, None, None, None]) # [batch_size, image_number, x坐标, y坐标, RGB三通道]
        with tf.name_scope("Labels"): # 定义Label的占位符
            self.Y_onehot = tf.placeholder(tf.float32, [None, 32, 32, 32, 2]) # [batch_size, x坐标, y坐标, z坐标，0/1的概率，真实值是0或1，预测值是0-1的概率]

        pp = preprocessor.Preprocessor(self.X) #对X(也就是Data)进行预处理
        self.X_preprocessed = pp.out_tensor # 处理过的X
        self.n_batchsize = tf.shape(self.X_preprocessed)[0] # n_batchsize取X的第0维

    def encoder(self): # 编码器
        print("encoder")
        if self.params["TRAIN"]["ENCODER_MODE"] == "RESIDUAL": # 如果encoder_mode超参为residual
            en = encoder.Residual_Encoder(self.X_preprocessed) # 使用residual的方法来encoder
        elif self.params["TRAIN"]["ENCODER_MODE"] == "ATTENTION": # 如果encoder_mode超参为Attention
            en = encoder.Residual_Attention_Encoder(self.X_preprocessed) # 就用attention_encoder
        self.encoder_output = en.out_tensor # 将encoder提取到的特征赋值给encoded_input，此时shape为[batch_size, image_num, 1024]

        # visualize transformation of input state to voxel
        if self.params["VIS"]["ENCODER_PROCESS"]:
            with tf.name_scope("misc"):
                feature_maps = tf.get_collection("feature_maps")
                fm_list = []
                for fm in feature_maps:
                    fm_slice = fm[0, 0, :, :, 0]
                    fm_shape = fm_slice.get_shape().as_list()
                    fm_slice = tf.pad(fm_slice, [[0, 0], [127-fm_shape[0], 0]])
                    fm_list.append(fm_slice)
                fm_img = tf.concat(fm_list, axis=0)
                tf.summary.image("feature_map_list", tf.expand_dims(
                    tf.expand_dims(fm_img, -1), 0))

    def recurrent_module(self): # 循环模块
        print("recurrent_module")
        with tf.name_scope("Recurrent_module"): # 在Recurrent_module名字域内
            rnn_mode = self.params["TRAIN"]["RNN_MODE"] # 读取超参训练时的RNN_MODE
            n_cell = self.params["TRAIN"]["RNN_CELL_NUM"] # 读取超参训练时的RNN_CELL_NUM
            n_hidden = self.params["TRAIN"]["RNN_HIDDEN_SIZE"] # 读取超参训练时的RNN_HIDDEN_SIZE

            # final block
            flat = encoder.flatten_sequence(self.encoder_output) # 将最后一个残差卷积块的输出平铺成一维向量 [batch_size, image_num, 2, 2, 256] -> [batch_size, image_num, 1024]
            fc0 = encoder.fully_connected_sequence(flat) # 全连接层 ->[batch_size, image_num, 1024]
            self.encoded_input = encoder.relu_sequence(fc0) # 全连接层的输出再经过relu激活函数

            n_timesteps = self.params["TRAIN"]["TIME_STEP_COUNT"] # 读取超参训练时的每次步数
            # feed a limited seqeuence of images
            if isinstance(n_timesteps, int) and n_timesteps > 0: # 如果n_timesteps是int类型且大于0
                for t in range(n_timesteps): # 循环n_timesteps次

                    if rnn_mode == "SIMPLE":
                        rnn = recurrent_module.Simple_Grid(initializer=self.init)
                    elif rnn_mode == "LSTM": # 如果读取的超参rnn_mode为LSTM
                        rnn = recurrent_module.LSTM_Grid(initializer=self.init) # 创建LSTM循环网络
                        hidden_state = (tf.zeros([self.n_batchsize, n_cell, n_cell, n_cell, n_hidden], name="zero_hidden_state"), tf.zeros(
                            [self.n_batchsize, n_cell, n_cell, n_cell, n_hidden], name="zero_cell_state")) # 创建初始的隐藏层，这里LSTM中的初始隐藏层包含隐藏层和细胞状态
                    else: # 如果读取的超参rnn_mode不为LSTM（希望使用GRU）
                        rnn = recurrent_module.GRU_Grid(initializer=self.init) # 创建GRU循环网络
                        hidden_state = tf.zeros(
                            [self.n_batchsize, n_cell, n_cell, n_cell, n_hidden], name="zero_hidden_state") # 创建初始的隐藏层

                    hidden_state = rnn.call(
                        self.encoded_input[:, t, :]) # 调用rnn.call()
                    if isinstance(hidden_state, tuple): # 如果隐藏状态是元组形式（LSTM）
                        hidden_state = hidden_state[0] # 取hidden_state中的隐藏层状态
                    hidden_state = tf.expand_dims(hidden_state, 1)

                    if t == 0:
                        self.n_hidden_state = hidden_state
                    else:    
                        self.n_hidden_state = tf.concat((self.n_hidden_state, hidden_state), 1)


            # else:  # feed an arbitray seqeuence of images,如果n_timesteps不满足条件，执行随机次数
            #     n_timesteps = tf.shape(self.X_preprocessed)[1]

            #     t = tf.constant(0)

            #     def condition(h, t): # condition()作为tf.while_loop()的判断条件
            #         return tf.less(t, n_timesteps) # 执行n_timesteps次

            #     def body(h, t): # body()作为tf.while_loop()的执行主体
            #         h = rnn.call(
            #             self.encoded_input[:, t, :], h) # 调用rnn.call()
            #         t = tf.add(t, 1) # 计数
            #         return h, t

            #     self.hidden_state, t = tf.while_loop( # tf.while_loop()的功能是在condition成立的时候执行body，第三个参数是给condition和body的参数
            #         condition, body, (self.hidden_state, t))

    # hidden_state [batch_size,n_cell,n_cell,n_cell,hidden_size]
    def decoder(self): # 解码器
        print("decoder")
        if self.params["TRAIN"]["DECODER_MODE"] == "DILATED": # 如果超参中设置的解码器模块是DILATED
            de = decoder.Dilated_Decoder(self.n_hidden_state) # 就用DILATED解码器
        elif self.params["TRAIN"]["DECODER_MODE"] == "RESIDUAL": # 如果超参中设置的解码器模块是RESIDUAL
            de = decoder.Residual_Decoder(self.n_hidden_state) # 就用RESIDUAL解码器
        else: # 如果都不是
            de = decoder.Simple_Decoder(self.n_hidden_state) # 就使用普通解码器
        self.logits = de.out_tensor # 保存解码器输出
        self.logits = tf.reduce_max(self.logits, 1)

        # visualize transformation of hidden state to voxel
        if self.params["VIS"]["DECODER_PROCESS"]:
            with tf.name_scope("misc"):
                feature_voxels = tf.get_collection("feature_voxels")
                fv_list = []
                for fv in feature_voxels:
                    fv_slice = fv[0, :, :, 0, 0]
                    fv_shape = fv_slice.get_shape().as_list()
                    fv_slice = tf.pad(fv_slice, [[0, 0], [32-fv_shape[0], 0]])
                    fv_list.append(fv_slice)
                fv_img = tf.concat(fv_list, axis=0)
                tf.summary.image("feature_voxel_list", tf.expand_dims(
                    tf.expand_dims(fv_img, -1), 0))

    def cal_loss(self):
        print("loss")
        voxel_loss = loss.Voxel_Softmax(self.Y_onehot, self.logits)
        self.loss = voxel_loss.loss
        self.softmax = voxel_loss.softmax
        tf.summary.scalar("loss", self.loss)

    def misc(self):
        print("misc")
        with tf.name_scope("misc"):
            self.step_count = tf.Variable(
                0, trainable=False, name="step_count")
            self.print = tf.Print(
                self.loss, [self.step_count, self.loss])

    def optimizer(self):
        print("optimizer")
        if self.params["TRAIN"]["OPTIMIZER"] == "ADAM":
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.params["TRAIN"]["ADAM_LEARN_RATE"], epsilon=self.params["TRAIN"]["ADAM_EPSILON"])
            tf.summary.scalar("adam_learning_rate", optimizer._lr)
        else:
            optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=self.params["TRAIN"]["GD_LEARN_RATE"])
            tf.summary.scalar("learning_rate", optimizer._learning_rate)

        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.apply_grad = optimizer.apply_gradients(
            grads_and_vars, global_step=self.step_count)

    def metric(self):
        print("metrics")
        with tf.name_scope("metrics"):
            Y = tf.argmax(self.Y_onehot, -1)
            predictions = tf.argmax(self.softmax, -1)
            acc, acc_op = tf.metrics.accuracy(Y, predictions)
            rms, rms_op = tf.metrics.root_mean_squared_error(
                self.Y_onehot, self.softmax)
            self.iou, self.iou_op = tf.metrics.mean_iou(Y, predictions, 2)
            self.metrics_op = tf.group(acc_op, rms_op, self.iou_op)
            self.print_iou = tf.Print(self.iou, ['iou', self.iou])

        tf.summary.scalar("accuracy", acc)
        tf.summary.scalar("rmse", rms)
        tf.summary.scalar("iou", self.iou)

        # initalize
        # config=tf.ConfigProto(log_device_placement=True)
        print("setup")
        self.summary_op = tf.summary.merge_all()
        self.sess = tf.InteractiveSession()
        if self.params["MODE"] == "DEBUG":
            self.sess = tf_debug.TensorBoardDebugWrapperSession(
                self.sess, "nat-oitwireless-inside-vapornet100-c-15126.Princeton.EDU:6064")

        # summaries
        print("summaries")
        if self.params["MODE"] == "TEST":
            self.test_writer = tf.summary.FileWriter(
                "{}\\test".format(self.MODEL_DIR), self.sess.graph)
        else:
            self.train_writer = tf.summary.FileWriter(
                "{}\\train".format(self.MODEL_DIR), self.sess.graph)
            self.val_writer = tf.summary.FileWriter(
                "{}\\val".format(self.MODEL_DIR), self.sess.graph)

        # initialize
        print("initialize")
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        print("...done!")

    def step(self, data, label, step_type):
        utils.make_dir(self.MODEL_DIR) # 如果目录文件夹不存在，就新建
        cur_dir = self.get_cur_epoch_dir() # 获取当前epoch存储路径
        data_npy, label_npy = utils.load_npy(data), utils.load_npy(label) # 读取之前存下来的数据和标签
        data_npy = data_npy[:, ::24]
        feed_dict = {self.X: data_npy, self.Y_onehot: label_npy} # 将数据放入feed_dict中

        if step_type == "train": # 如果是在训练
            fetches = [self.apply_grad, self.loss, self.summary_op,
                       self.print, self.step_count, self.metrics_op]
            out = self.sess.run(fetches, feed_dict)
            loss, summary, step_count = out[1], out[2], out[4]

            self.train_writer.add_summary(summary, global_step=step_count)
        elif step_type == "debug":
            fetchs = [self.apply_grad]
            options = tf.RunOptions(trace_level=3)
            run_metadata = tf.RunMetadata()
            out = self.sess.run(fetches, feed_dict,
                                options=options, run_metadata=run_metadata)
        else:
            fetchs = [self.softmax, self.loss, self.summary_op, self.print_iou,
                      self.step_count, self.metrics_op]
            out = self.sess.run(fetchs, feed_dict)
            softmax, loss, summary, step_count = out[0], out[1], out[2], out[4]

            if step_type == "val":
                self.val_writer.add_summary(summary, global_step=step_count)
            elif step_type == "test":
                self.test_writer.add_summary(summary, global_step=step_count)

            # display the result of each element of the validation batch
            if self.params["VIS"]["VALIDATION_STEP"]:
                i = random.randint(0, len(data_npy)-1)
                x, y, yp = data_npy[i], label_npy[i], softmax[i]
                name = "{}\\{}_{}".format(cur_dir, step_count,
                                         utils.get_file_name(data[i])[0:-2])
                vis.img_sequence(x, "{}_x.png".format(name))
                vis.voxel_binary(y, "{}_y.png".format(name))
                vis.voxel_binary(yp, "{}_yp.png".format(name))

        return loss

    def save(self): # 用于保存模型
        cur_dir = self.get_cur_epoch_dir()
        epoch_name = utils.grep_epoch_name(cur_dir)
        model_builder = tf.saved_model.builder.SavedModelBuilder(
            cur_dir + "\\model")
        model_builder.add_meta_graph_and_variables(self.sess, [epoch_name])
        model_builder.save()

    def predict(self, x):
        return self.sess.run([self.softmax], {self.X: x})

    def get_params(self):
        utils.make_dir(self.MODEL_DIR) # 如果路径不存在，就创建路径
        with open(self.MODEL_DIR+"\\params.json") as fp: # 打开json文件
            return json.load(fp) # 读取json文件

    def create_epoch_dir(self):
        cur_ind = self.epoch_index()
        save_dir = os.path.join(self.MODEL_DIR, "epoch_{}".format(cur_ind+1))
        utils.make_dir(save_dir)
        return save_dir

    def get_cur_epoch_dir(self): # 用于得到最近的一个epoch的路径
        cur_ind = self.epoch_index()
        save_dir = os.path.join(
            self.MODEL_DIR, "epoch_{}".format(cur_ind))
        return save_dir

    def epoch_index(self):
        return utils.get_latest_epoch_index(self.MODEL_DIR) # 找到最近的一个epoch的索引


class Network_restored:
    def __init__(self, model_dir):
        if "epoch" not in model_dir: # 如果model_dir中不包含'epoch'
            model_dir = utils.get_latest_epoch(model_dir) # 就将model_dir重新赋值为最近的一个epoch的路径

        epoch_name = utils.grep_epoch_name(model_dir) # 得到epoch_*(*是数字)
        self.sess = tf.Session(graph=tf.Graph())
        tf.saved_model.loader.load( # 读取模型参数
            self.sess, [epoch_name], model_dir + "\\model")

    def predict(self, x, in_name="Data/Placeholder:0", sm_name="Loss_Voxel_Softmax/clip_by_value:0"):
        if x.ndim == 4: # 如果x的维度只有4个，也就是没有batch_size这一维
            x = np.expand_dims(x, 0) # 就给x这个数据加上batch_size这一维

        in_tensor = self.sess.graph.get_tensor_by_name(in_name)
        softmax = self.sess.graph.get_tensor_by_name(sm_name)
        return self.sess.run(softmax, {in_tensor: x})

    def get_operations(self):
        return self.sess.graph.get_operations()

    def get_closest_tensor(self, name, ndim):
        op_list = self.get_operations() # tf.Graph.get_operations()返回图中的操作节点列表
        for op in op_list: # 遍历图中的所有操作节点
            try:
                n = len(op.inputs[0].shape)
                if name in op.name and n == ndim:
                    ret = op.name+":0"
                    print(ret)
                    return ret
            except:
                pass

    def feature_maps(self, x):
        pass
