import math
import os
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, r2_score
from time import time
import argparse
import LoadData as DATA
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
import logging
import math

from numpy.random import seed
# from tensorflow import set_random_seed

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True  # self-adaption


def parse_args():
    parser = argparse.ArgumentParser(description="Run CFFM.")
    parser.add_argument('--path', nargs='?', default='data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='frappe',
                        help='Choose a dataset.')
    parser.add_argument('--epoch', type=int, default=50,
                        help='Number of epochs.')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='flag for pretrain. 1: initialize from pretrain; 0: randomly initialize; -1: save the model to pretrain file')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')
    parser.add_argument('--inner_dims', type=int, default=32,
                        help='Number of inner dimensions.')
    parser.add_argument('--outer_dims', type=int, default=32,
                        help='Number of outer dimensions.')
    parser.add_argument('--lamda', type=float, default=0,
                        help='Regularizer for bilinear part.')
    parser.add_argument('--keep', nargs='?', default='[1.0,1.0]',
                        help='Keep probility (1-dropout) of each layer. 1: no dropout. The first index is for the attention-aware pairwise interaction layer.')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate.')
    parser.add_argument('--loss_type', nargs='?', default='square_loss',
                        help='Specify a loss type (square_loss or log_loss or mse or hybrid or crossentropy).')
    parser.add_argument('--optimizer', nargs='?', default='AdagradOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show the results per X epochs (0, 1 ... any positive integer)')
    parser.add_argument('--batch_norm', type=int, default=0,
                        help='Whether to perform batch normaization (0 disable or 1 enable)')
    parser.add_argument('--tensorboard', type=int, default=0,
                        help='Whether to log record of tensorboard (0 disable or 1 enable)')
    parser.add_argument('--num_field', type=int, default=3,
                        help='Valid dimension of the dataset. (e.g. frappe=10, ml-tag=3, book-crossing=6)')
    parser.add_argument('--linear_att', type=int, default=1,
                        help='Control the randomness by scaling linear attention part (0 disable or 1 enable)')
    parser.add_argument('--att_dim', type=int, default=0,
                        help='Dimension of linear attention (0 is the same as num_field, otherwise integers are greater than 0')
    parser.add_argument('--lamda_att', type=float, default=1.0,
                        help='Control the randomness by scaling linear attention part')

    parser.add_argument('--inner_conv', type=int, default=1,
                        help='Where open inner convolution part (0 disable or 1 enable)')
    parser.add_argument('--gamma_inner', type=int, default=1.0,
                        help='Control the weight of inner convolution component')

    parser.add_argument('--outer_conv', type=int, default=1,
                        help='Where open outer convolution part (0 disable or 1 enable)')
    parser.add_argument('--beta_outer', type=int, default=1.0,
                        help='Control the weight of outer convolution component')

    parser.add_argument('--activation', nargs='?', default='relu',
                        help='Choose an activation function. (relu, prelu, elu, selu, gelu)')

    return parser.parse_args()


def configure_logging(logFilename):
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s:%(message)s',  # define format
        datefmt='%Y-%m-%d %A %H:%M:%S',  # define time
        filename=logFilename,  # define log name
        filemode='a')  # define writing pattern 'w' or 'a'
    # Define a Handler and set a format which output to console
    console = logging.StreamHandler()  # console handler
    console.setLevel(logging.INFO)  # handler level
    formatter = logging.Formatter('%(asctime)s %(filename)s:%(message)s')  # define handler format
    console.setFormatter(formatter)
    # Create an instance
    logging.getLogger().addHandler(console)  # instantiate handler


class CFFM:
    def __init__(self, features_M, pretrain_flag, save_file, inner_dims, outer_dims, loss_type, epoch, batch_size, learning_rate,
                 lamda_bilinear, keep, optimizer_type, batch_norm, verbose, tensorboard, num_field, linear_att, att_dim,
                 lamda_att,inner_conv,gamma_inner,outer_conv,beta_outer,activation_function,
                 random_seed=2021):
        # bind params to class
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.inner_dims = inner_dims
        self.outer_dims = outer_dims
        self.pretrain_flag = pretrain_flag
        self.save_file = save_file
        self.loss_type = loss_type
        self.features_M = features_M
        self.lamda_bilinear = lamda_bilinear
        self.keep = keep
        self.epoch = epoch
        self.random_seed = random_seed
        self.optimizer_type = optimizer_type
        self.batch_norm = batch_norm
        self.verbose = verbose
        self.tensorboard = tensorboard
        self.num_field = num_field
        self.linear_att = linear_att
        if att_dim == 0:
            self.att_dim = num_field
        else:
            self.att_dim = att_dim
        self.lamda_att = lamda_att
        self.inner_conv = inner_conv
        self.gamma_inner = gamma_inner
        self.outer_conv = outer_conv
        self.beta_outer = beta_outer
        self.num_interactions = int(self.num_field * (self.num_field - 1) / 2)

        if activation_function == 'relu':
            self.activation = tf.nn.relu
        elif activation_function == 'elu':
            self.activation = tf.nn.elu
        elif activation_function == 'prelu':
            self.activation = self.prelu
        elif activation_function == 'selu':
            self.activation = tf.nn.selu
        elif activation_function == 'gelu':
            self.activation = self.gelu

        #  create_save_folder
        self.create_save_folder(save_file)

        self.train_rmse, self.valid_rmse, self.test_rmse = [], [], []
        self.train_r2, self.valid_r2, self.test_r2 = [], [], []

    def gelu(self,input_tensor):
        cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
        return input_tensor * cdf

    def prelu(self,data):
        alpha = 0.25
        return tf.nn.relu(data) + tf.multiply(alpha, -tf.nn.relu(-data))

    def train(self, data):
        init_op = self.build_graph()
        self.saver = tf.train.Saver()
        with tf.Session(config=tf_config) as self.sess:
            self.sess.run(init_op)

            # merged
            if self.tensorboard > 0:
                merged = tf.summary.merge_all()
                writer = tf.summary.FileWriter("logs/", self.sess.graph)

            self.calculate_parameters()
            if self.verbose > 0:
                t2 = time()
                init_train_rmse, init_train_r2 = self.evaluate(data.Train_data)
                init_valid_rmse, init_validation_r2 = self.evaluate(data.Validation_data)
                init_test_rmse, init_test_r2 = self.evaluate(data.Test_data)
                logging.info(("Init_RMSE: train=%.4f,validation=%"
                              ".4f,test=%.4f | Init_R2: train=%.4f,validation=%"
                              ".4f,test=%.4f [%.1f s] " % (
                                  init_train_rmse, init_valid_rmse, init_test_rmse, init_train_r2, init_validation_r2,
                                  init_test_r2,
                                  time() - t2)))

            for epoch in range(self.epoch):
                t1 = time()
                data.Train_data['X'], data.Train_data['Y'] = self.shuffle_in_unison_scary(data.Train_data['X'],
                                                                                          data.Train_data['Y'])
                total_batch = int(len(data.Train_data['Y']) / self.batch_size)
                for i in range(total_batch):
                    # generate a batch
                    batch_xs = self.get_random_block_from_data(data.Train_data, self.batch_size)
                    # Fit training
                    feed_dict = {self.train_features: batch_xs['X'], self.train_labels: batch_xs['Y'],
                                 self.dropout_keep: self.keep, self.train_phase: True}
                    # enable tensorboard will be time-consuming in each epoch
                    if self.tensorboard > 0:
                        summary, loss, opt, test_dimension1, test_dimension2 = self.sess.run(
                            (merged, self.loss, self.optimizer, self.test_dimension1, self.test_dimension2,self.test_dimension3, self.test_dimension4),
                            feed_dict=feed_dict)

                        writer.add_summary(summary, epoch * total_batch + i)
                    else:
                        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)


                t2 = time()
                # output validation
                train_rmse, train_r2 = self.evaluate(data.Train_data)
                valid_rmse, valid_r2 = self.evaluate(data.Validation_data)
                test_rmse, test_r2 = self.evaluate(data.Test_data)

                self.train_rmse.append(train_rmse)
                self.valid_rmse.append(valid_rmse)
                self.test_rmse.append(test_rmse)

                self.train_r2.append(train_r2)
                self.valid_r2.append(valid_r2)
                self.test_r2.append(test_r2)

                if self.verbose > 0 and epoch % self.verbose == 0:
                    logging.info((
                        "Epoch %d [%.1f s] RMSE: train=%.4f,validation=%.4f,Test=%.4f | R2: train=%.4f,validation=%.4f,Test=%.4f [%.1f s]"
                        % (epoch + 1, t2 - t1, train_rmse, valid_rmse, test_rmse, train_r2,
                           valid_r2, test_r2, time() - t2)))

                if self.eva_termination(self.valid_rmse):
                    break

                if self.pretrain_flag < 0:
                    logging.info("Save model to file as pretrain.")
                    self.saver.save(self.sess, self.save_file)

    def create_placeholders(self):
        # None * features_M
        self.train_features = tf.placeholder(tf.int32, shape=[None, None],
                                             name="train_features_fm")
        # None * 1
        self.train_labels = tf.placeholder(tf.float32, shape=[None, 1], name="train_labels_fm")
        self.dropout_keep = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_afm")
        self.train_phase = tf.placeholder(tf.bool, name="train_phase_fm")

    def initialize_variables(self):
        self.weights = dict()
        if self.pretrain_flag > 0:
            weight_saver = tf.train.import_meta_graph(self.save_file + '.meta')
            pretrain_graph = tf.get_default_graph()
            feature_embeddings = pretrain_graph.get_tensor_by_name('feature_embeddings:0')
            feature_bias = pretrain_graph.get_tensor_by_name('feature_bias:0')
            bias = pretrain_graph.get_tensor_by_name('bias:0')
            with tf.Session() as sess:
                weight_saver.restore(sess, self.save_file)
                fe, fb, b = sess.run([feature_embeddings, feature_bias, bias])
            self.weights['feature_embeddings'] = tf.Variable(fe, dtype=tf.float32)
            self.weights['feature_bias'] = tf.Variable(fb, dtype=tf.float32)
            self.weights['bias'] = tf.Variable(b, dtype=tf.float32)
        else:

            if self.inner_conv == 1:
                # features_M * K
                self.weights['inner_embeddings'] = tf.Variable(
                    tf.random_normal([self.features_M, self.inner_dims], 0.0, 0.1),
                    name='inner_embeddings')
            #     seed=2021

            if self.outer_conv == 1:
                # features_M * K
                self.weights['outer_embeddings'] = tf.Variable(
                    tf.random_normal([self.features_M, self.outer_dims], 0.0, 0.01),
                    name='outer_embeddings')
                # , seed = 2023
                # interaction * 1
                # self.outer_W = self.weight_variable([self.num_interactions, 1])
                # self.outer_b = self.weight_variable([1])
                self.weights['outer_W'] = self.weight_variable([self.num_interactions, 1])
                self.weights['outer_b'] = self.weight_variable([1])


            # features_M * 1
            self.weights['feature_bias'] = tf.Variable(
                tf.random_normal([self.features_M, 1], 0.0, 0.0), name='feature_bias')
            # , seed = 2019
            if self.linear_att == 1:
                # num_field * num_field
                self.weights['bias_W'] = self.weight_variable([self.att_dim, self.att_dim])
                self.weights['bias_b'] = self.weight_variable([self.att_dim])
            # 1
            self.weights['bias'] = tf.Variable(tf.constant(0.0), name='bias')



        # tensorboard
        if self.tensorboard > 0:
            tf.summary.histogram('inner_embeddings', self.weights['inner_embeddings'])
            tf.summary.histogram('outer_embeddings', self.weights['outer_embeddings'])
            tf.summary.histogram('feature_bias', self.weights['feature_bias'])
            tf.summary.histogram('bias', self.weights['bias'])


    def create_inference_convolutional_feature_interaction_FM(self):

        add_component = list()

        # inner convolution component
        if self.inner_conv == 1:
            self.inner = []
            self.feature_embeddings = tf.nn.embedding_lookup(self.weights['inner_embeddings'], self.train_features)
            for i in range(0, self.num_field):
                for j in range(i + 1, self.num_field):
                    self.feature_i = self.feature_embeddings[:, i, :]
                    self.feature_j = self.feature_embeddings[:, j, :]

                    # None * K
                    self.inner.append(tf.multiply(self.feature_i, self.feature_j))

            # interaction * None * K
            self.inner_input = tf.stack(self.inner)
            # None * interaction * K
            self.inner_input = tf.transpose(self.inner_input, perm=[1, 0, 2])
            # None * interaction * K * 1
            self.inner_input = tf.expand_dims(self.inner_input, -1)

            self.inner_input = self.activation(self.inner_input)



            (self.weights['inner_layer_conv_weight_0'], self.weights['inner_layer_conv_bias_0']) = self.conv_weight(1, 2, 1, 2)



            self.inner_conv1 = self.conv_layer(self.inner_input, (self.weights['inner_layer_conv_weight_0'], self.weights['inner_layer_conv_bias_0']), strides=[1, 1, 2, 1],
                                               padding='VALID')

            self.inner_conv1 = self.activation(self.inner_conv1)
            self.inner_conv1_ = self.max_pool_layer(self.inner_input, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1])
            self.inner_conv1 = tf.add(self.inner_conv1,self.inner_conv1_)
            self.inner_max = tf.reshape(tf.squeeze(self.inner_conv1),shape=[-1,int(self.num_interactions*16*2)])
            self.test_dimension2 = self.inner_max

            # for i in range(1):
            #     self.inner_max = tf.layers.dense(self.inner_max, units=int(self.num_interactions),activation=None)
            for i in range(1):
                self.inner_max = tf.layers.dense(self.inner_max, units=1,activation=None)

            self.final2 = self.inner_max

            add_component.append(self.final2)



        # outer convlution component
        if self.outer_conv==1:

            outer_ = []
            self.sum_pooling = []

            # # None * interaction * F
            self.outer_embeddings = tf.nn.embedding_lookup(self.weights['outer_embeddings'], self.train_features)
            for i in range(0, self.num_field):
                for j in range(i + 1, self.num_field):
                    # None * F * 1
                    self.feature_i_ = tf.expand_dims(self.outer_embeddings[:, i, :], -1)
                    # None * 1 * F
                    self.feature_j_ = tf.transpose(tf.expand_dims(self.outer_embeddings[:, j, :], -1), perm=[0, 2, 1])
                    # None * F * F
                    outer_.append(tf.multiply(self.feature_i_, self.feature_j_))

            # # num_field * None * F * F
            self.outer_input = tf.stack(outer_)
            # None * num_field * F * F
            self.self_conv = tf.transpose(self.outer_input, perm=[1, 2, 3, 0])
            # None * num_field * F * 1

            # self.outer_layer = []
            # self.weights['outer_layer'] = []
            # # [filter_height, filter_width, in_channels, out_channels]]
            self.conv_depth = int(math.log(self.outer_dims,2))
            print('conv_depth:',self.conv_depth)
            for i in range(self.conv_depth):
                (self.weights['outer_layer_conv_weight_'+str(i)],self.weights['outer_layer_conv_bias_'+str(i)]) \
                    = self.conv_weight(2, 2, self.num_interactions, self.num_interactions)



            self.sum_pooling.append(tf.reduce_sum(self.self_conv,axis=[2,3]))

            alpha = 0.25
            for i in range(self.conv_depth):
                self.self_conv = self.conv_layer(self.self_conv, (self.weights['outer_layer_conv_weight_'+str(i)],self.weights['outer_layer_conv_bias_'+str(i)]),
                                                 strides=[1, 2, 2, 1],padding='VALID')
                self.self_conv = self.activation(self.self_conv)
                # self.self_conv = tf.nn.relu(self.self_conv)

                self.test_dimension1 = tf.reduce_sum(self.self_conv,axis=[2,3])
                self.sum_pooling.append(self.test_dimension1)


            self.t1 = self.sum_pooling[0]
            for i in range(1,self.conv_depth):
                self.t1 = tf.concat([self.t1, self.sum_pooling[i]],axis=1)


            # self.self_conv = tf.squeeze(self.self_conv)
            self.test_dimension2 = self.t1
            # self.test_dimension2 = self.self_conv

            # full connected layer
            # for i in range(1):
            #     self.final = tf.layers.dense(inputs=self.self_conv, units=1)

            # (None * num_interaction) * （num_interaction * 1) -> None * 1

            self.t1 = tf.layers.dense(self.t1, units=32, activation=None)
            self.final = tf.layers.dense(self.t1, units=1, activation=None)

            # self.final = tf.tensordot(self.self_conv, self.weights['outer_W'],axes=1) + self.weights['outer_b']
            # self.test_dimension3 = self.final
            self.final = tf.reshape(self.beta_outer * self.final,shape=[-1,1])
            # self.test_dimension4 = self.final
            # self.final = tf.reshape(self.final, shape=[-1, 1])

            add_component.append(self.final)


        # None * num_field * 1
        self.feature_bias = tf.nn.embedding_lookup(self.weights['feature_bias'], self.train_features)
        if self.linear_att == 1:
            # None * num_field
            self.feature_bias = tf.squeeze(self.feature_bias)
            # self.linear = tf.matmul(self.feature_bias, self.weights['bias_W_2']) + self.weights['bias_b_2']
            # self.linear = tf.matmul(self.linear, self.weights['bias_W_1']) + self.weights['bias_b_1']
            # self.linear = tf.reshape(self.linear,shape=[-1,self.num_field])


            # None * num_field
            self.linear = tf.matmul(self.feature_bias,self.weights['bias_W']) + self.weights['bias_b']
            # None * num_field
            self.linear = self.linear/self.lamda_att
            # None * num_field
            self.linear = tf.nn.softmax(self.linear,name="linear_attention")
            # attention (None * num_field)
            self.linear = tf.reshape(tf.multiply(self.feature_bias,self.linear),shape=[-1,self.num_field])
            # None * 1
            for i in range(1):
                self.linear = tf.layers.dense(inputs=self.linear, units=1, activation=None)
        else:
            # None * num_field * 1
            self.linear = tf.reduce_sum(self.feature_bias, axis=1)

        add_component.append(self.linear)

        # None * 1
        self.bias = self.weights['bias'] * tf.ones_like(self.train_labels)
        add_component.append(self.bias)


        self.out = tf.add_n([add_component[i] for i in range(len(add_component))], name='out')





    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=1)
        # seed = 2022
        # initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)


    def conv_weight(self, filter_height, filter_width, in_channels, out_channels):
        # [filter_height, filter_width, in_channels, out_channels]]
        return (self.weight_variable([filter_height, filter_width, in_channels, out_channels]),
                self.bias_variable([out_channels]))

    def conv_layer(self, input, P, strides,padding):  # P:   P[0]: W;   P[1]:b  default:strides=[1, 1, 1, 1]
        conv = tf.nn.conv2d(input, P[0], strides=strides,
                            padding=padding)  # strides = [batch= 1, height, width, channels=1]
        return tf.nn.relu(conv + P[1])  # bias_add and activate

    def max_pool_layer(self, x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='VALID'):
        # stride [1, x_movement, y_movement, 1]
        # [1, height, width, 1]
        # [1, stride,stride, 1]
        return tf.nn.max_pool(x, ksize=ksize, strides=strides, padding=padding)

    def create_loss(self):
        if self.loss_type == 'square_loss':
            if self.lamda_bilinear > 0:
                self.loss = tf.nn.l2_loss(tf.subtract(self.train_labels, self.out)) + tf.contrib.layers.l2_regularizer(
                    self.lamda_bilinear)(self.weights['inner_embeddings']) + tf.contrib.layers.l2_regularizer(
                    self.lamda_att)(self.weights['outer_embeddings'])  # regulizer
            else:
                self.loss = tf.sqrt(tf.reduce_mean(tf.square(self.train_labels - self.out)) + 1e-10)

        elif self.loss_type == 'log_loss':
            self.out = tf.sigmoid(self.out)
            if self.lamda_bilinear > 0:
                self.loss = tf.contrib.losses.log_loss(self.out, self.train_labels, weights=1.0, epsilon=1e-07,
                                                       scope=None) + tf.contrib.layers.l2_regularizer(
                    self.lamda_bilinear)(self.weights['feature_embeddings']) + tf.contrib.layers.l2_regularizer(
                    self.lamda_att)(self.weights['attention_W'])
            else:
                self.loss = tf.contrib.losses.log_loss(self.out, self.train_labels, weights=1.0, epsilon=1e-07,
                                                       scope=None)
        elif self.loss_type == 'mse':
            self.loss = tf.reduce_mean(tf.square(self.train_labels - self.out))
        elif self.loss_type == 'mae':
            self.loss = tf.reduce_mean(tf.abs(self.train_labels - self.out))

        elif self.loss_type == 'hybrid':
            self.loss = 0.5 * tf.nn.l2_loss(tf.subtract(self.train_labels, self.out)) + \
                        0.5 * tf.contrib.losses.log_loss(self.out, self.train_labels, weights=1.0, epsilon=1e-07,
                                                         scope=None)
        tf.summary.scalar('loss', self.loss)


    def create_optimizer(self):
        # Optimizer.
        if self.optimizer_type == 'AdamOptimizer':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                    epsilon=1e-8).minimize(self.loss)
        elif self.optimizer_type == 'AdagradOptimizer':
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                       initial_accumulator_value=1e-8).minimize(self.loss)
        elif self.optimizer_type == 'GradientDescentOptimizer':
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        elif self.optimizer_type == 'MomentumOptimizer':
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                self.loss)

    def build_graph(self):
        # seed(self.random_seed)
        # set_random_seed(self.random_seed)

        self.create_placeholders()
        self.initialize_variables()
        self.create_inference_convolutional_feature_interaction_FM()
        self.create_loss()
        self.create_optimizer()
        init_op = tf.global_variables_initializer()
        return init_op

    def calculate_parameters(self):
        # number of params
        total_parameters = 0
        for variable in list(self.weights.values()):
            shape = variable.get_shape()  # shape is an array of tf.Dimension
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        if self.verbose > 0:
            logging.info("#params: %d" % total_parameters)

    # shuffle two lists simutaneously
    def shuffle_in_unison_scary(self, x, y):
        x_, y_ = shuffle(x, y, random_state=self.random_seed)
        return x_, y_

    def get_random_block_from_data(self, data, batch_size):  # generate a random block of training data
        start_index = np.random.randint(0, len(data['Y']) - batch_size)
        X, Y = [], []
        # forward get sample
        i = start_index
        while len(X) < batch_size and i < len(data['X']):
            if len(data['X'][i]) == len(data['X'][start_index]):
                Y.append([data['Y'][i]])
                X.append(data['X'][i])
                i = i + 1
            else:
                break
        # backward get sample
        i = start_index
        while len(X) < batch_size and i >= 0:
            if len(data['X'][i]) == len(data['X'][start_index]):
                Y.append([data['Y'][i]])
                X.append(data['X'][i])
                i = i - 1
            else:
                break
        return {'X': X, 'Y': Y}

    def evaluate(self, data):  # evaluate the results for an input set
        num_example = len(data['Y'])
        # fetch the first batch
        batch_index = 0
        batch_xs = self.get_ordered_block_from_data(data, self.batch_size, batch_index)
        # batch_xs = data
        y_pred = None
        # if len(batch_xs['X']) > 0:
        while len(batch_xs['X']) > 0:
            num_batch = len(batch_xs['Y'])
            feed_dict = {self.train_features: batch_xs['X'], self.train_labels: [[y] for y in batch_xs['Y']],
                         self.dropout_keep: list(1.0 for i in range(len(self.keep))), self.train_phase: False}
            # a_out, batch_out = self.sess.run((self.attention_out, self.out), feed_dict=feed_dict)
            batch_out = self.sess.run((self.out), feed_dict=feed_dict)

            if batch_index == 0:
                y_pred = np.reshape(batch_out, (num_batch,))
            # eliminate one dimension
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))
            # fetch the next batch
            batch_index += 1
            batch_xs = self.get_ordered_block_from_data(data, self.batch_size, batch_index)

        y_true = np.reshape(data['Y'], (num_example,))

        predictions_bounded = np.maximum(y_pred, np.ones(num_example) * min(y_true))  # bound the lower values
        predictions_bounded = np.minimum(predictions_bounded,
                                         np.ones(num_example) * max(y_true))  # bound the higher values
        RMSE = math.sqrt(mean_squared_error(y_true, predictions_bounded))

        R2 = r2_score(y_true, predictions_bounded)
        return RMSE, R2

    def get_ordered_block_from_data(self, data, batch_size, index):  # generate a ordered block of data
        start_index = index * batch_size
        X, Y = [], []
        # get sample
        i = start_index
        while len(X) < batch_size and i < len(data['X']):
            if len(data['X'][i]) == len(data['X'][start_index]):
                Y.append(data['Y'][i])
                X.append(data['X'][i])
                i = i + 1
            else:
                break
        return {'X': X, 'Y': Y}

    def eva_termination(self, valid):
        if len(valid) > 5:
            if valid[-1] > valid[-2] and valid[-2] > valid[-3] and valid[-3] > valid[-4] and valid[-4] > valid[-5]:
                return True
        return False

    def create_save_folder(self, save_file):
        if not os.path.exists(save_file):
            os.makedirs(save_file)

    def batch_norm_layer(self, x, train_phase):
        # Note: the decay parameter is tunable
        bn_train = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
                              is_training=True, trainable=True)
        bn_inference = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
                                  is_training=False, trainable=True)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

if __name__ == '__main__':
    args = parse_args()

    # configure logging
    logFilename = 'logging.log'
    configure_logging(logFilename)

    if args.verbose > 0:
        logging.info(
            "CFFM: dataset=%s, factors=%d, loss_type=%s, #epoch=%d, batch=%d, lr=%.4f, lambda=%.1e, keep=%s, optimizer=%s"
            ", batch_norm=%d, num_field=%d, linear_att=%d, att_dim=%d,lamda_att=%.2f,inner_conv=%d,gamma_inner=%.1f,outer_conv=%d,"
            "beta_outer=%.1f, activation=%s"
            % (args.dataset, args.inner_dims, args.loss_type, args.epoch, args.batch_size, args.lr, args.lamda,
               eval(args.keep), args.optimizer, args.batch_norm,args.num_field,args.linear_att,args.att_dim,args.lamda_att,
               args.inner_conv,args.gamma_inner,args.outer_conv,args.beta_outer,args.activation))


    # Data loading
    data = DATA.LoadData(args.path, args.dataset, args.loss_type)
    save_file = 'pretrain/CFFM/%s_%d/%s_%d' % (args.dataset, args.inner_dims, args.dataset, args.inner_dims)

    # Training
    t1 = time()

    cf_fm = CFFM(data.features_M, args.pretrain, save_file, args.inner_dims,args.outer_dims, args.loss_type, args.epoch,
                  args.batch_size, args.lr, args.lamda, eval(args.keep), args.optimizer, args.batch_norm, args.verbose,
                  args.tensorboard, args.num_field,args.linear_att, args.att_dim, args.lamda_att,args.inner_conv,
                  args.gamma_inner,args.outer_conv,args.beta_outer,args.activation)

    cf_fm.train(data)

    # choice the best RMSE
    best_valid_score = min(cf_fm.valid_rmse)
    best_epoch = cf_fm.valid_rmse.index(best_valid_score)
    logging.info("Best Iter of RMSE (validation)= %d train = %.4f, valid = %.4f, test = %.4f [%.1f s]"
                 % (best_epoch + 1, cf_fm.train_rmse[best_epoch], cf_fm.valid_rmse[best_epoch],
                    cf_fm.test_rmse[best_epoch],
                    time() - t1))

    # choice the best R2 score
    best_valid_r2 = max(cf_fm.valid_r2)
    best_valid_r2 = cf_fm.valid_r2.index(best_valid_r2)
    logging.info("Best Iter of R2 (validation)= %d train = %.4f, valid = %.4f, test = %.4f [%.1f s]"
                 % (best_epoch + 1, cf_fm.train_r2[best_valid_r2], cf_fm.valid_r2[best_valid_r2],
                    cf_fm.test_r2[best_valid_r2],
                    time() - t1))


