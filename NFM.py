import math
import os
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, r2_score
from time import time
import argparse
import LoadData as DATA
# from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
import logging

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True  # self-adaption


def parse_args():
    parser = argparse.ArgumentParser(description="Run FM.")
    parser.add_argument('--path', nargs='?', default='data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='book-crossing',
                        help='Choose a dataset.')
    parser.add_argument('--epoch', type=int, default=500,
                        help='Number of epochs.')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='flag for pretrain. 1: initialize from pretrain; 0: randomly initialize; -1: save the model to pretrain file')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors.')
    parser.add_argument('--lamda', type=float, default=0,
                        help='Regularizer for bilinear part.')
    parser.add_argument('--keep_prob', type=float, default=1,
                        help='Keep probility (1-dropout_ratio) for the Bi-Interaction layer. 1: no dropout')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate.')
    parser.add_argument('--loss_type', nargs='?', default='square_loss',
                        help='Specify a loss type (square_loss or log_loss or hybrid or mse).')
    parser.add_argument('--optimizer', nargs='?', default='GradientDescentOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show the results per X epochs (0, 1 ... any positive integer)')
    parser.add_argument('--batch_norm', type=int, default=0,
                        help='Whether to perform batch normaization (0 disable or 1 enable)')
    parser.add_argument('--tensorboard', type=int, default=0,
                        help='Whether to log record of tensorboard (0 disable or 1 enable)')
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


class NFM:
    def __init__(self, features_M, pretrain_flag, save_file, hidden_factor, loss_type, epoch, batch_size, learning_rate,
                 lamda_bilinear, keep,optimizer_type, batch_norm, verbose, tensorboard, random_seed=2019):
        # bind params to class
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_factor = hidden_factor
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
        #  create_save_folder
        self.create_save_folder(save_file)
        self.train_rmse, self.valid_rmse, self.test_rmse = [], [], []
        self.train_r2, self.valid_r2, self.test_r2 = [], [], []

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
                data.Train_data['X'], data.Train_data['Y'] = self.shuffle_in_unison_scary(data.Train_data['X'], data.Train_data['Y'])
                total_batch = int(len(data.Train_data['Y']) / self.batch_size)
                for i in range(total_batch):
                    # generate a batch
                    batch_xs = self.get_random_block_from_data(data.Train_data, self.batch_size)
                    # Fit training
                    feed_dict = {self.train_features: batch_xs['X'], self.train_labels: batch_xs['Y'],
                                 self.dropout_keep: self.keep, self.train_phase: True}
                    # enable tensorboard will be time-consuming in each epoch
                    if self.tensorboard > 0:
                        summary, loss, opt = self.sess.run((merged, self.loss, self.optimizer), feed_dict=feed_dict)
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
        self.dropout_keep = tf.placeholder(tf.float32, name="dropout_keep_fm")
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
            # features_M * K
            self.weights['feature_embeddings'] = tf.Variable(
                tf.random_normal([self.features_M, self.hidden_factor], 0.0, 0.01),
                name='feature_embeddings')
            # features_M * 1
            self.weights['feature_bias'] = tf.Variable(
                tf.random_uniform([self.features_M, 1], 0.0, 0.0), name='feature_bias')
            # 1
            self.weights['bias'] = tf.Variable(tf.constant(0.0), name='bias')

        # tensorboard
        if self.tensorboard > 0:
            tf.summary.histogram('feature_embeddings', self.weights['feature_embeddings'])
            tf.summary.histogram('feature_bias', self.weights['feature_bias'])
            tf.summary.histogram('bias', self.weights['bias'])



    def create_inference_FM(self):
        # get the summed up embeddings of features.
        # None * M * K
        self.nonzero_embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.train_features)

        # None * 1 * K
        self.summed_features_emb = tf.reduce_sum(self.nonzero_embeddings, axis=1, keep_dims=True)
        # get the element-multiplication
        # None * 1 * K
        self.summed_features_emb_square = tf.square(self.summed_features_emb)

        # square_sum part
        # None * M * K
        self.squared_features_emb = tf.square(self.nonzero_embeddings)
        # None * 1 * K
        self.squared_summed_features_emb = tf.reduce_sum(self.squared_features_emb, axis=1, keep_dims=True)

        # FM operation
        # None * 1 * K
        self.FM = 0.5 * tf.subtract(self.summed_features_emb_square, self.squared_summed_features_emb, name='FM')
        # if self.batch_norm:
        #     self.FM = self.batch_norm_layer(self.FM, train_phase=self.train_phase, scope_bn='bn_fm')
        # dropout at the FM layer
        self.FM = tf.nn.dropout(self.FM, self.dropout_keep)
        # None * K
        self.FM_OUT = tf.reduce_sum(self.FM, 1, name="fm_out")
        # dropout at the FM layer
        # None * K
        self.FM_OUT = tf.nn.dropout(self.FM_OUT, self.dropout_keep)

        # out part
        # None * 1
        Bilinear = tf.reduce_sum(self.FM_OUT, 1, keep_dims=True)
        # None * features_M * 1 -> None * 1 * 1
        Feature_bias = tf.reduce_sum(tf.nn.embedding_lookup(self.weights['feature_bias'], self.train_features), axis=1)
        # None * 1
        Bais = self.weights['bias'] * tf.ones_like(self.train_labels)
        self.out = tf.add_n([Bilinear, Feature_bias, Bais], name='out')

    def create_loss(self):
        if self.loss_type == 'square_loss':
            if self.lamda_bilinear > 0:
                self.loss = tf.nn.l2_loss(tf.subtract(self.train_labels, self.out)) + tf.contrib.layers.l2_regularizer(
                    self.lamda_bilinear)(self.weights['feature_embeddings']) # regulizer
            else:
                self.loss = tf.sqrt(tf.reduce_mean(tf.square(self.train_labels - self.out)) + 1e-10)
                # self.loss = tf.nn.l2_loss(tf.subtract(self.train_labels, self.out))

        elif self.loss_type == 'log_loss':
            self.out = tf.sigmoid(self.out)
            if self.lamda_bilinear > 0:
                self.loss = tf.contrib.losses.log_loss(self.out, self.train_labels, weights=1.0, epsilon=1e-07,
                                                       scope=None) + tf.contrib.layers.l2_regularizer(
                    self.lamda_bilinear)(self.weights['feature_embeddings'])
            else:
                self.loss = tf.contrib.losses.log_loss(self.out, self.train_labels, weights=1.0, epsilon=1e-07,
                                                       scope=None)
        elif self.loss_type == 'mse':
            self.loss = tf.reduce_mean(tf.square(self.train_labels - self.out))

        elif self.loss_type == 'hybrid':
            self.loss = tf.nn.l2_loss(tf.subtract(self.train_labels, self.out)) + \
                        tf.contrib.losses.log_loss(self.out, self.train_labels, weights=1.0, epsilon=1e-07,
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
        tf.set_random_seed(self.random_seed)
        self.create_placeholders()
        self.initialize_variables()
        self.create_inference_FM()
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

    # shuffle two lists simutaneously
    def shuffle_in_unison_scary(self, x, y):
        x_, y_ = shuffle(x, y)
        return x_, y_

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
                         self.dropout_keep: 1.0, self.train_phase: False}
            # a_out, batch_out = self.sess.run((self.attention_out, self.out), feed_dict=feed_dict)
            predictions = self.sess.run((self.out), feed_dict=feed_dict)
            if batch_index == 0:
                y_pred = np.reshape(predictions, (num_batch,))
            # eliminate one dimension
            else:
                y_pred = np.concatenate((y_pred, np.reshape(predictions, (num_batch,))))
            # fetch the next batch
            batch_index += 1
            batch_xs = self.get_ordered_block_from_data(data, self.batch_size, batch_index)

        y_true = np.reshape(data['Y'], (num_example,))

        # For example:
        # if the predicted result is -1.5, while the true value is -1, we are supposed to bound the result to -1.
        predictions_bounded = np.maximum(y_pred, np.ones(num_example) * min(y_true))  # bound the lower values
        predictions_bounded = np.minimum(predictions_bounded,
                                         np.ones(num_example) * max(y_true))  # bound the higher values
        # print(y_true)
        # print(y_pred)
        RMSE = math.sqrt(mean_squared_error(y_true, predictions_bounded))
        R2 = r2_score(y_true, predictions_bounded)
        return RMSE,R2

    def eva_termination(self, valid):
        if len(valid) > 5:
            if valid[-1] > valid[-2] and valid[-2] > valid[-3] and valid[-3] > valid[-4] and valid[-4] > valid[-5]:
                return True
        return False

    def create_save_folder(self, save_file):
        if not os.path.exists(save_file):
            os.makedirs(save_file)


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

    # def batch_norm_layer(self, x, train_phase, scope_bn):
    #     bn = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
    #                           is_training=train_phase, reuse=None, trainable=True, scope=scope_bn)
    #     return bn

if __name__ == '__main__':
    args = parse_args()

    # configure logging
    logFilename = 'logging.log'
    configure_logging(logFilename)

    if args.verbose > 0:
        print(
            "NFM: dataset=%s, factors=%d, loss_type=%s, #epoch=%d, batch=%d, lr=%.4f, lambda=%.1e, keep=%.2f, optimizer=%s, batch_norm=%d"
            % (args.dataset, args.hidden_factor, args.loss_type, args.epoch, args.batch_size, args.lr, args.lamda,
               args.keep_prob, args.optimizer, args.batch_norm))

    # data = DATA.LoadData(args.path, args.dataset)

    # Data loading
    data = DATA.LoadData(args.path, args.dataset, args.loss_type)
    save_file = 'pretrain/NFM/%s_%d/%s_%d' % (args.dataset, args.hidden_factor, args.dataset, args.hidden_factor)

    # Training
    t1 = time()
    nfm = NFM(data.features_M, args.pretrain, save_file, args.hidden_factor, args.loss_type, args.epoch,
                   args.batch_size, args.lr, args.lamda, args.keep_prob, args.optimizer, args.batch_norm,
                   args.verbose,args.tensorboard)
    nfm.train(data)

    # choice the best RMSE
    best_valid_score = min(nfm.valid_rmse)
    best_epoch = nfm.valid_rmse.index(best_valid_score)
    logging.info("Best Iter of RMSE (validation)= %d train = %.4f, valid = %.4f, test = %.4f [%.1f s]"
                 % (best_epoch + 1, nfm.train_rmse[best_epoch], nfm.valid_rmse[best_epoch],
                    nfm.test_rmse[best_epoch],
                    time() - t1))

    # choice the best R2 score
    best_valid_r2 = max(nfm.valid_r2)
    best_valid_r2 = nfm.valid_r2.index(best_valid_r2)
    logging.info("Best Iter of R2 (validation)= %d train = %.4f, valid = %.4f, test = %.4f [%.1f s]"
                 % (best_epoch + 1, nfm.train_r2[best_valid_r2], nfm.valid_r2[best_valid_r2],
                    nfm.test_r2[best_valid_r2],
                    time() - t1))


