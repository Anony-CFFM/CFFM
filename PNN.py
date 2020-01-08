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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True  # self-adaption


def parse_args():
    parser = argparse.ArgumentParser(description="Run FM.")
    parser.add_argument('--path', nargs='?', default='data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-tag',
                        help='Choose a dataset.')
    parser.add_argument('--epoch', type=int, default=300,
                        help='Number of epochs.')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='flag for pretrain. 1: initialize from pretrain; 0: randomly initialize; -1: save the model to pretrain file')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors.')
    parser.add_argument('--lamda', type=float, default=0,
                        help='Regularizer for bilinear part.')
    parser.add_argument('--keep', nargs='?', default='[1.0,0.5]',
                        help='Keep probility (1-dropout) of each layer. 1: no dropout. The first index is for the attention-aware pairwise interaction layer.')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate.')
    parser.add_argument('--loss_type', nargs='?', default='square_loss',
                        help='Specify a loss type (square_loss or log_loss).')
    parser.add_argument('--optimizer', nargs='?', default='AdagradOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show the results per X epochs (0, 1 ... any positive integer)')
    parser.add_argument('--batch_norm', type=int, default=0,
                        help='Whether to perform batch normaization (0 disable or 1 enable)')
    parser.add_argument('--tensorboard', type=int, default=0,
                        help='Whether to log record of tensorboard (0 disable or 1 enable)')
    parser.add_argument('--num_field', type=int, default=3,
                        help='Valid dimension of the dataset. (e.g. frappe=10, ml-tag=3)')
    parser.add_argument('--att_dim', type=int, default=128,
                        help='dimension of attention')
    parser.add_argument('--lamda_attention', type=float, default=0.5,
                        help='Regularizer for attention part.')
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


class PNN:
    def __init__(self, features_M, pretrain_flag, save_file, hidden_factor, loss_type, epoch, batch_size, learning_rate,
                 lamda_bilinear, keep, optimizer_type, batch_norm, verbose, tensorboard, num_field, att_dim, lamda_attention,
                 random_seed=2016):
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
        self.num_field = num_field
        self.att_dim = att_dim
        self.lamda_attention = lamda_attention

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
            # features_M * K
            self.weights['feature_embeddings'] = tf.Variable(
                tf.random_normal([self.features_M, self.hidden_factor], 0.0, 0.01),
                name='feature_embeddings')
            # features_M * 1
            self.weights['feature_bias'] = tf.Variable(
                tf.random_uniform([self.features_M, 1], 0.0, 0.0), name='feature_bias')
            # 1
            self.weights['bias'] = tf.Variable(tf.constant(0.0), name='bias')

            # attention part
            glorot = np.sqrt(2.0 / (self.hidden_factor + self.att_dim))
            # K * A
            self.weights['attention_W'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(self.hidden_factor, self.att_dim)),
                dtype=tf.float32, name="attention_W")
            # 1 * A
            self.weights['attention_b'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.att_dim)), dtype=tf.float32, name="attention_b")
            #  A
            self.weights['attention_p'] = tf.Variable(np.random.normal(loc=0, scale=1, size=(self.att_dim)), dtype=tf.float32, name="attention_p")
            # K * 1
            self.weights['prediction'] = tf.Variable(np.ones((self.hidden_factor, 1)), dtype=tf.float32, name='prediction')


        # tensorboard
        if self.tensorboard > 0:
            tf.summary.histogram('feature_embeddings', self.weights['feature_embeddings'])
            tf.summary.histogram('feature_bias', self.weights['feature_bias'])
            tf.summary.histogram('bias', self.weights['bias'])


    def create_inference_PNN(self):
        self.deep_layers = [32, 32]
        self.deep_init_size = 50
        self.dropout_keep_deep = [0.5, 0.5, 0.5]
        self.deep_layer_activation = tf.nn.relu
        self.batch_norm = 0
        self.batch_norm_decay = 0.995
        self.use_inner = True
        self.deep_layers_activation = tf.nn.relu


        # Product Layers
        if self.use_inner:
            self.weights['product-quadratic-inner'] = tf.Variable(
                tf.random_normal([self.deep_init_size, self.num_field], 0.0, 0.01))
        else:
            self.weights['product-quadratic-outer'] = tf.Variable(
                tf.random_normal([self.deep_init_size, self.hidden_factor, self.hidden_factor], 0.0, 0.01))

        self.weights['product-linear'] = tf.Variable(
            tf.random_normal([self.deep_init_size, self.num_field, self.hidden_factor], 0.0, 0.01))
        self.weights['product-bias'] = tf.Variable(tf.random_normal([self.deep_init_size, ], 0, 0, 1.0))
        # deep layers
        num_layer = len(self.deep_layers)
        input_size = self.deep_init_size
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

        glorot = np.sqrt(2.0 / (input_size + 1))
        self.weights['output'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[-1], 1)),
                                        dtype=np.float32)
        self.weights['output_bias'] = tf.Variable(tf.constant(0.01), dtype=np.float32)

        # Embeddings
        self.embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.train_features)  # N * M * K

        # Linear Singal
        linear_output = []
        for i in range(self.deep_init_size):
            linear_output.append(tf.reshape(
                tf.reduce_sum(tf.multiply(self.embeddings, self.weights['product-linear'][i]), axis=[1, 2]),
                shape=(-1, 1)))  # N * 1

        self.lz = tf.concat(linear_output, axis=1)  # N * init_deep_size

        # Quardatic Singal
        quadratic_output = []
        if self.use_inner:
            for i in range(self.deep_init_size):
                theta = tf.multiply(self.embeddings,
                                    tf.reshape(self.weights['product-quadratic-inner'][i], (1, -1, 1)))  # N * F * K
                quadratic_output.append(
                    tf.reshape(tf.norm(tf.reduce_sum(theta, axis=1), axis=1), shape=(-1, 1)))  # N * 1

        else:
            embedding_sum = tf.reduce_sum(self.embeddings, axis=1)
            p = tf.matmul(tf.expand_dims(embedding_sum, 2), tf.expand_dims(embedding_sum, 1))  # N * K * K
            for i in range(self.deep_init_size):
                theta = tf.multiply(p, tf.expand_dims(self.weights['product-quadratic-outer'][i], 0))  # N * K * K
                quadratic_output.append(tf.reshape(tf.reduce_sum(theta, axis=[1, 2]), shape=(-1, 1)))  # N * 1

        self.lp = tf.concat(quadratic_output, axis=1)  # N * init_deep_size
        # add -> None * deep_init_size
        # add -> None * deep_init_size
        # relu -> None * deep_init_size
        self.y_deep = tf.nn.relu(tf.add(tf.add(self.lz, self.lp), self.weights['product-bias']))
        self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[0])

        # Deep component
        for i in range(0, len(self.deep_layers)):
            self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights["layer_%d" % i]), self.weights["bias_%d" % i])
            self.y_deep = self.deep_layers_activation(self.y_deep)
            self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[i + 1])

        self.out = tf.add(tf.matmul(self.y_deep, self.weights['output']), self.weights['output_bias'])

    def create_loss(self):
        if self.loss_type == 'square_loss':
            if self.lamda_bilinear > 0:
                self.loss = tf.nn.l2_loss(tf.subtract(self.train_labels, self.out)) + tf.contrib.layers.l2_regularizer(
                    self.lamda_bilinear)(self.weights['feature_embeddings']) + tf.contrib.layers.l2_regularizer(
                    self.lamda_att)(self.weights['attention_W'])  # regulizer
            else:
                # self.loss = tf.nn.l2_loss(tf.subtract(self.train_labels, self.out))
                self.loss = tf.sqrt(tf.reduce_mean(tf.square(self.train_labels - self.out)) + 1e-10)

        elif self.loss_type == 'log_loss':
            self.out = tf.sigmoid(self.out)
            if self.lamda_bilinear > 0:
                self.loss = tf.contrib.losses.log_loss(self.out, self.train_labels, weights=1.0, epsilon=1e-07,
                                                       scope=None) + tf.contrib.layers.l2_regularizer(
                    self.lamda_bilinear)(self.weights['feature_embeddings']) + tf.contrib.layers.l2_regularizer(
                    self.lamda_att)(self.weights['attention_W'])
            else:
                self.loss = tf.sqrt(tf.reduce_mean(tf.square(self.train_labels - self.out)) + 1e-10)

                # self.loss = tf.contrib.losses.log_loss(self.out, self.train_labels, weights=1.0, epsilon=1e-07,
                #                                        scope=None)
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
        tf.set_random_seed(self.random_seed)
        self.create_placeholders()
        self.initialize_variables()
        # self.create_inference_test_FM()
        self.create_inference_PNN()
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

    if args.verbose > 0:
        print(
            "PNN: dataset=%s, factors=%d, loss_type=%s, #epoch=%d, batch=%d, lr=%.4f, lambda=%.1e, keep=%s, optimizer=%s, batch_norm=%d"
            % (args.dataset, args.hidden_factor, args.loss_type, args.epoch, args.batch_size, args.lr, args.lamda,
               eval(args.keep), args.optimizer, args.batch_norm))

    # data = DATA.LoadData(args.path, args.dataset)

    # Data loading
    data = DATA.LoadData(args.path, args.dataset, args.loss_type)
    save_file = 'pretrain/PNN/%s_%d/%s_%d' % (args.dataset, args.hidden_factor, args.dataset, args.hidden_factor)

    # Training
    t1 = time()
    model = PNN(data.features_M, args.pretrain, save_file, args.hidden_factor, args.loss_type, args.epoch,
                   args.batch_size, args.lr, args.lamda, eval(args.keep), args.optimizer, args.batch_norm,args.verbose,
                   args.tensorboard,args.num_field, args.att_dim,args.lamda_attention)
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
    logging.info(model.total_parameters)
