import os
import sys
import math
import numpy as np
import tensorflow as tf
from lib import utils


class Simple_Grid:
    def __init__(self, n_cells=4, n_input=1024, n_hidden_state=128, initializer=None):
        with tf.name_scope("SIMPLE_Grid"):
            if initializer is None:
                init = tf.contrib.layers.xavier_initializer()
            else:
                init = initializer
            
            # parameters for the hidden state
            self.W = Weight_Matrices(
                n_cells, n_input, n_hidden_state, initializer=init)
            self.b = tf.Variable(init(
                [n_cells, n_cells, n_cells, n_hidden_state]), name="b")

    def pre_activity(self, W, x, b):
        return W.multiply_grid(x) + b

    def call(self, input_tensor):
        h_t = tf.nn.relu(self.pre_activity(self.W, input_tensor, self.b))
        return h_t


class GRU_Grid:
    def __init__(self,  n_cells=4,  n_input=1024, n_hidden_state=128, initializer=None):
        with tf.name_scope("GRU_Grid"):
            if initializer is None:
                init = tf.contrib.layers.xavier_initializer()
            else:
                init = initializer

            # parameters for the hidden state and update & reset gates
            self.W = [Weight_Matrices(
                n_cells, n_input, n_hidden_state, initializer=init)]*3
            self.U = [tf.Variable(init(
                [3, 3, 3, n_hidden_state, n_hidden_state]), name="U")]*3
            self.b = [tf.Variable(init(
                [n_cells, n_cells, n_cells, n_hidden_state]), name="b")]*3

            params = utils.read_params()
            if params["VIS"]["HISTOGRAMS"]:
                for i in range(3):
                    tf.summary.histogram("U[{}]".format(i), self.U[i])
                    tf.summary.histogram("b[{}]".format(i), self.b[i])

    def pre_activity(self, W, x, U, h, b):
        return W.multiply_grid(x) + tf.nn.conv3d(h, U, strides=[1, 1, 1, 1, 1], padding="SAME") + b

    def call(self, input_tensor, prev_hidden):
        # update gate
        u_t = tf.sigmoid(
            self.pre_activity(self.W[0], input_tensor, self.U[0], prev_hidden, self.b[0]))
        # reset gate
        r_t = tf.sigmoid(
            self.pre_activity(self.W[1], input_tensor, self.U[1], prev_hidden,  self.b[1]))

        # hidden state
        h_t_1 = (1 - u_t) * prev_hidden
        h_t_2 = u_t * tf.tanh(self.pre_activity(self.W[2], input_tensor,
                                                self.U[2], r_t * prev_hidden, self.b[2]))
        h_t = h_t_1 + h_t_2
        return h_t


class LSTM_Grid:
    def __init__(self, n_cells=4, n_input=1024, n_hidden_state=128, initializer=None):
        with tf.name_scope("LSTM_Grid"):
            if initializer is None: # 如果没有给定初始化器的参数
                init = tf.contrib.layers.xavier_initializer() # 使用xavier初始化器
            else: # 反之，如果给定了初始化器的参数
                init = initializer # 使用给定的初始化器

            # 定义forget gate，input gate，output gate,cell state的参数
            self.W = [Weight_Matrices(
                n_cells, n_input, n_hidden_state, initializer=init)]*4
            self.U = [tf.Variable(init(
                [3, 3, 3, n_hidden_state, n_hidden_state]), name="U")]*4
            self.b = [tf.Variable(init(
                [n_cells, n_cells, n_cells, n_hidden_state]), name="b")]*4

            params = utils.read_params() # 读取超参
            if params["VIS"]["HISTOGRAMS"]: # 如果需要可视化直方图
                for i in range(4):
                    tf.summary.histogram("U[{}]".format(i), self.U[i])
                    tf.summary.histogram("b[{}]".format(i), self.b[i])

    def pre_activity(self, W, x, U, h, b):
        # tf.nn.conv3d()的第一个参数是输入，结构为：[batch, in_depth, in_height, in_width, in_channels]
        # tf.nn.conv3d()的第二个参数是卷积核，结构与输入相同，第三个参数是步幅
        return W.multiply_grid(x) + tf.nn.conv3d(h, U, strides=[1, 1, 1, 1, 1], padding="SAME") + b

    def call(self, input_tensor, prev_state):
        prev_hidden_state, prev_cell_state = prev_state # LSTM定义时的hidden_state里包含hidden_state和cell_state

        # forget gate
        f_t = tf.sigmoid(
            self.pre_activity(self.W[0], input_tensor, self.U[0], prev_hidden_state, self.b[0]))

        # input gate
        i_t = tf.sigmoid(
            self.pre_activity(self.W[1], input_tensor, self.U[1], prev_hidden_state,  self.b[1]))

        # output gate
        o_t = tf.sigmoid(
            self.pre_activity(self.W[2], input_tensor, self.U[2], prev_hidden_state, self.b[2]))

        # cell state
        s_t_1 = f_t * prev_cell_state
        s_t_2 = i_t * tf.tanh(self.pre_activity(self.W[3], input_tensor,
                                                self.U[3], prev_hidden_state, self.b[3]))
        s_t = s_t_1 + s_t_2
        h_t = o_t*tf.tanh(s_t)

        return (h_t, s_t)


class Weight_Matrices:
    def __init__(self,  n_cells, n_x, n_h,  initializer=None):
        with tf.name_scope("Weight_Matrices"):
            params = utils.read_params() # 读取超参
            # class variables
            self.n_cells = n_cells # RNN_cell_num

            if initializer is None: # 如果没有设定初始化方式
                init = tf.contrib.layers.xavier_initializer() # 就用Xavier初始化
            else: # 反之如果设定了初始化方式
                init = initializer # 就用设定的初始化方式

            # 创建x_list的结构：[n_cells,n_cells,n_cells][n_x,n_h]
            with tf.name_scope("x_list"):
                x_list = []
                for x in range(self.n_cells):
                    with tf.name_scope("y_list"):
                        y_list = []
                        for y in range(self.n_cells):
                            z_list = []
                            with tf.name_scope("z_list"):
                                for z in range(self.n_cells):
                                    name = "W_{}{}{}".format(x, y, z)
                                    W = tf.Variable(init(
                                        [n_x, n_h]), name=name)

                                    if params["VIS"]["HISTOGRAMS"]:
                                        tf.summary.histogram(name, W)
                                    z_list.append(W)
                            y_list.append(z_list)
                    x_list.append(y_list)

            self.weight_matrix_grid = x_list

    # multiply each of weight matrix with x
    def multiply_grid(self, x):
        with tf.name_scope("multiply_grid"):
            x_list = []
            for i in range(self.n_cells):
                y_list = []
                for j in range(self.n_cells):
                    z_list = []
                    for k in range(self.n_cells):
                        transformed_vector = tf.matmul( # tf.matmul()的功能是进行矩阵乘法
                            x, self.weight_matrix_grid[i][j][k])
                        z_list.append(transformed_vector)
                    y_list.append(z_list)
                x_list.append(y_list)

        # tf.convert_to_tensor()用于将不同数据变成张量
        # tf.transpose()用于交换张量不同维度的位置
        return tf.transpose(tf.convert_to_tensor(x_list), [3, 0, 1, 2, 4]) 