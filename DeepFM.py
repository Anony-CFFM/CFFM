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

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True  # self-adaption


def parse_args():
    parser = argparse.ArgumentParser(description="Run DeepCross.")
    parser.add_argument('--path', nargs='?', default='data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='frappe',
                        help='Choose a dataset.')
    parser.add_argument('--epoch', type=int, default=300,
                        help='Number of epochs.')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='flag for pretrain. 1: initialize from pretrain; 0: randomly initialize; -1: save the model to pretrain file')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors.')
    parser.add_argument('--l2_reg', type=float, default=0.0,
                        help='Regularizer for bilinear part.')
    parser.add_argument('--drop_out_keep', nargs='?', default='[0.5,0.5,0.5]',
                        help='Keep probility of deep layer.')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate.')
    parser.add_argument('--loss_type', nargs='?', default='mse',
                        help='Specify a loss type (mse or logloss).')
    parser.add_argument('--optimizer', nargs='?', default='AdagradOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show the results per X epochs (0, 1 ... any positive integer)')
    parser.add_argument('--batch_norm', type=int, default=0,
                        help='Whether to perform batch normaization (0 disable or 1 enable)')
    parser.add_argument('--num_field', type=int, default=10,
                        help='Valid dimension of the dataset. (e.g. frappe=10, ml-tag=3)')
    parser.add_argument('--dropout_fm', nargs='?', default='[1.0,1.0]',
                        help='Keep probility of FM layer.')
    parser.add_argument('--deep_layers', nargs='?', default='[128,128]',
                        help='Each number of hidden dimension in deep layers.')
    parser.add_argument('--activation', nargs='?', default='relu',
                    help='Which activation function to use for deep layers: relu, sigmoid, tanh, identity')
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


class DeepFM:
    def __init__(self, features_M, pretrain_flag, save_file, hidden_factor, loss_type, epoch, batch_size, learning_rate,
                 l2_reg, drop_out_keep, optimizer_type, batch_norm, verbose, num_field, dropout_fm,
                 deep_layers,activation, random_seed=2019):
        # bind params to class
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_factor = hidden_factor
        self.pretrain_flag = pretrain_flag
        self.save_file = save_file
        self.loss_type = loss_type
        self.features_M = features_M
        self.l2_reg = l2_reg
        self.drop_out_keep = drop_out_keep
        self.epoch = epoch
        self.random_seed = random_seed
        self.optimizer_type = optimizer_type
        self.batch_norm = batch_norm
        self.verbose = verbose
        self.num_field = num_field
        self.dropout_fm = dropout_fm
        self.deep_layers = deep_layers
        self.activation = activation


        #  create_save_folder
        self.create_save_folder(save_file)

        self.train_rmse, self.valid_rmse, self.test_rmse = [], [], []
        self.train_r2, self.valid_r2, self.test_r2 = [], [], []

    def train(self, data):
        init = self.build_graph()
        self.saver = tf.train.Saver()
        with tf.Session(config=tf_config) as self.sess:
            self.sess.run(init)


            self.calculate_parameters()
            if self.verbose > 0:
                t2 = time()
                init_train_rmse, init_train_r2 = self.evaluate(data.Train_data)
                init_valid_rmse, init_validation_r2 = self.evaluate(data.Validation_data)
                init_test_rmse, init_test_r2 = self.evaluate(data.Test_data)
                logging.info(("Init_RMSE: train=%.4f,validation=%"
                              ".4f,test=%.4f | Init_R2: train=%.4f,validation=%"
                              ".4f,test=%.4f [%.1f s] " % (
                    init_train_rmse, init_valid_rmse, init_test_rmse, init_train_r2, init_validation_r2, init_test_r2,
                                  time() - t2)))

            for epoch in range(self.epoch):
                t1 = time()
                data.Train_data['X'], data.Train_data['Y'] = self.shuffle_in_unison_scary(data.Train_data['X'], data.Train_data['Y'])
                total_batch = int(len(data.Train_data['Y']) / self.batch_size)
                for i in range(total_batch):
                    # generate a batch
                    batch_xs = self.get_random_block_from_data(data.Train_data, self.batch_size)
                    # Fit training
                    feed_dict = {self.train_features: batch_xs['X'], self.train_labels: batch_xs['Y'],
                                 self.dropout_keep: self.drop_out_keep,
                                 self.dropout_keep_fm: self.dropout_fm,
                                 self.train_phase: True}

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
        self.train_labels = tf.placeholder(tf.float32, shape=[None, 1], name="train_labels_DeepCross")
        self.dropout_keep = tf.placeholder(tf.float32, shape=[None], name="dropout_keep")
        self.dropout_keep_fm = tf.placeholder(tf.float32, shape=[None], name='dropout_keep_fm')
        self.train_phase = tf.placeholder(tf.bool, name="train_phase_fm")

    def initialize_variables(self):
        self.use_fm = True
        self.use_deep = True
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
            # features_M * K
            self.weights['feature_embeddings'] = tf.Variable(
                tf.random_normal([self.features_M, self.hidden_factor], 0.0, 0.01),
                name='feature_embeddings')
            # features_M * 1
            self.weights['feature_bias'] = tf.Variable(
                tf.random_uniform([self.features_M, 1], 0.0, 0.0), name='feature_bias')

            # deep layers
            num_layer = len(self.deep_layers)
            #
            input_size = self.num_field * self.hidden_factor
            glorot = np.sqrt(2.0 / (input_size + self.deep_layers[0]))

            self.weights['layer_0'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(input_size, self.deep_layers[0])), dtype=np.float32
            )
            self.weights['bias_0'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[0])), dtype=np.float32
            )

            for i in range(1, num_layer):
                glorot = np.sqrt(2.0 / (self.deep_layers[i - 1] + self.deep_layers[i]))
                self.weights["layer_%d" % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i - 1], self.deep_layers[i])),
                    dtype=np.float32)  # layers[i-1] * layers[i]
                self.weights["bias_%d" % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])),
                    dtype=np.float32)  # 1 * layer[i]

            # final concat projection layer

            if self.use_fm and self.use_deep:
                input_size = self.num_field + self.hidden_factor + self.deep_layers[-1]
            elif self.use_fm:
                input_size = self.num_field + self.hidden_factor
            elif self.use_deep:
                input_size = self.deep_layers[-1]

            glorot = np.sqrt(2.0 / (input_size + 1))
            self.weights['concat_projection'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(input_size, 1)),
                                                            dtype=np.float32)
            self.weights['concat_bias'] = tf.Variable(tf.constant(0.01), dtype=np.float32)



    def create_inference_DeepFM(self):

        # Embeddings
        self.embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.train_features)  # N * M * K

        # first order term
        # None * M * 1
        self.y_first_order = tf.nn.embedding_lookup(self.weights['feature_bias'], self.train_features)
        # None * M
        self.y_first_order = tf.reduce_sum(self.y_first_order, axis=2)
        self.y_first_order = tf.nn.dropout(self.y_first_order, self.dropout_keep_fm[0])

        # second order term
        # sum-square-part
        self.summed_features_emb = tf.reduce_sum(self.embeddings, 1)  # None * k
        self.summed_features_emb_square = tf.square(self.summed_features_emb)  # None * K

        # squre-sum-part
        self.squared_features_emb = tf.square(self.embeddings)
        self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1)  # None * K

        # second order
        self.y_second_order = 0.5 * tf.subtract(self.summed_features_emb_square, self.squared_sum_features_emb)
        # None * K
        self.y_second_order = tf.nn.dropout(self.y_second_order, self.dropout_keep_fm[1])

        # Deep component
        # None * (M * K)
        self.y_deep = tf.reshape(self.embeddings, shape=[-1, self.num_field * self.hidden_factor])
        self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep[0])

        # None * layer[i]
        for i in range(0, len(self.deep_layers)):
            self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights["layer_%d" % i]), self.weights["bias_%d" % i])
            self.y_deep = self.activation_function(self.y_deep)
            self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep[i + 1])

        # ----DeepFM---------
        if self.use_fm and self.use_deep:
            concat_input = tf.concat([self.y_first_order, self.y_second_order, self.y_deep], axis=1)
        elif self.use_fm:
            concat_input = tf.concat([self.y_first_order, self.y_second_order], axis=1)
        elif self.use_deep:
            concat_input = self.y_deep
        self.out = tf.add(tf.matmul(concat_input, self.weights['concat_projection']), self.weights['concat_bias'])




    def create_loss(self):
        # loss
        if self.loss_type == "logloss":
            self.out = tf.nn.sigmoid(self.out)
            self.loss = tf.losses.log_loss(self.train_labels, self.out)
        elif self.loss_type == "mse":
            self.loss = tf.nn.l2_loss(tf.subtract(self.train_labels, self.out))
        # l2 regularization on weights
        if self.l2_reg > 0:
            self.loss += tf.contrib.layers.l2_regularizer(
                self.l2_reg)(self.weights["concat_projection"])
            if self.use_deep:
                for i in range(len(self.deep_layers)):
                    self.loss += tf.contrib.layers.l2_regularizer(
                        self.l2_reg)(self.weights["layer_%d" % i])


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

    def create_activation_function(self):
        if args.activation == 'relu':
            self.activation_function = tf.nn.relu
        if args.activation == 'sigmoid':
            self.activation_function = tf.sigmoid
        elif args.activation == 'tanh':
            self.activation_function == tf.tanh
        elif args.activation == 'identity':
            self.activation_function = tf.identity

    def build_graph(self):
        tf.set_random_seed(self.random_seed)
        self.create_placeholders()
        self.initialize_variables()
        self.create_activation_function()
        self.create_inference_DeepFM()
        self.create_loss()
        self.create_optimizer()
        init = tf.global_variables_initializer()
        return init

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

        self.total_parameters = total_parameters

    # shuffle two lists simutaneously
    def shuffle_in_unison_scary(self, x, y):
        x_, y_ = shuffle(x, y)
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
                         self.dropout_keep: list(1.0 for i in range(len(self.drop_out_keep))),
                         self.dropout_keep_fm: list(1.0 for i in range(len(self.dropout_fm))),
                         self.train_phase: False}
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

        R2 = r2_score(y_true, y_pred)
        return RMSE, R2

    def get_ordered_block_from_data(self, data, batch_size, index):  # generate a ordered block of data
        start_index = index*batch_size
        X , Y = [], []
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

    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
                              is_training=train_phase, reuse=None, trainable=True, scope=scope_bn)
        return bn

if __name__ == '__main__':
    args = parse_args()

    # configure logging
    logFilename = 'logging.log'
    configure_logging(logFilename)


    # Data loading
    data = DATA.LoadData(args.path, args.dataset, args.loss_type)
    save_file = 'pretrain/DeepFM/%s_%d/%s_%d' % (args.dataset, args.hidden_factor, args.dataset, args.hidden_factor)

    if args.verbose > 0:
        logging.info('DeepFM:dataset=%s,pretrain=%d,save_file=%s,hidden_factor=%d,loss_type=%s,#epoch=%d, batch=%d,lr=%.4f,'
                     'l2_reg=%.1e,drop_out_keep=%s,optimizer=%s, batch_norm=%d,verbose=%s,num_field=%d,dropout_fm=%s,deep_layers=%s,'
                     'activation=%s' % (args.dataset, args.pretrain, save_file, args.hidden_factor, args.loss_type, args.epoch,
                   args.batch_size, args.lr, args.l2_reg, args.drop_out_keep, args.optimizer, args.batch_norm,args.verbose,
                      args.num_field, args.dropout_fm, args.deep_layers,args.activation))


    # Training
    t1 = time()


    model = DeepFM(data.features_M, args.pretrain, save_file, args.hidden_factor, args.loss_type, args.epoch,
                   args.batch_size, args.lr, args.l2_reg, eval(args.drop_out_keep), args.optimizer, args.batch_norm,args.verbose,
                      args.num_field, eval(args.dropout_fm), eval(args.deep_layers),args.activation)
    model.train(data)

    # choice the best RMSE
    best_valid_score = min(model.valid_rmse)
    best_epoch = model.valid_rmse.index(best_valid_score)
    logging.info("Best Iter of RMSE (validation)= %d train = %.4f, valid = %.4f, test = %.4f [%.1f s]"
                 % (best_epoch + 1, model.train_rmse[best_epoch], model.valid_rmse[best_epoch],
                    model.test_rmse[best_epoch],
                    time() - t1))

    # choice the best R2 score
    best_valid_r2 = max(model.valid_r2)
    best_valid_r2 = model.valid_r2.index(best_valid_r2)
    logging.info("Best Iter of R2 (validation)= %d train = %.4f, valid = %.4f, test = %.4f [%.1f s]"
                 % (best_epoch + 1, model.train_r2[best_valid_r2], model.valid_r2[best_valid_r2],
                    model.test_r2[best_valid_r2],
                    time() - t1))
