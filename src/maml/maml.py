""" Code for the MAML algorithm and network definitions. """
from __future__ import print_function
import numpy as np
import sys
import tensorflow as tf
try:
    import special_grads
except KeyError as e:
    print('WARN: Cannot define MaxPoolGrad, likely already defined for this version of tensorflow: %s' % e,
          file=sys.stderr)

from .utils import mse, xent, conv_block, normalize

class MAML:
    def __init__(self, dim_input=1, dim_output=1, test_num_updates=5, update_lr=0.2, meta_lr=0.2):
        """ must call construct_model() after initializing MAML! """
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.update_lr = update_lr
        self.meta_lr = tf.placeholder_with_default(meta_lr, ())
        self.test_num_updates = test_num_updates

        self.dim_hidden = [40, 40]
        self.loss_func = mse
        self.forward = self.forward_fc
        self.construct_weights = self.construct_fc_weights


    def construct_model(self, input_tensors=None,
                        prefix='metatrain_',
                        num_updates=10,
                        metatrain_iterations=10,
                        meta_batch_size =20):
        # a: training data for inner gradient, b: test data for meta gradient
        if input_tensors is None:
            self.inputa = tf.placeholder(tf.float32)
            self.inputb = tf.placeholder(tf.float32)
            self.labela = tf.placeholder(tf.float32)
            self.labelb = tf.placeholder(tf.float32)
        else:
            self.inputa = input_tensors['inputa']
            self.inputb = input_tensors['inputb']
            self.labela = input_tensors['labela']
            self.labelb = input_tensors['labelb']

        with tf.variable_scope('model', reuse=None) as training_scope:
            if 'weights' in dir(self):
                training_scope.reuse_variables()
                weights = self.weights
            else:
                # Define the weights
                self.weights = weights = self.construct_weights()

            # outputbs[i] and lossesb[i] is the output and loss after i+1 gradient updates
            lossesa, outputas, lossesb, outputbs = [], [], [], []
            accuraciesa, accuraciesb = [], []
            num_updates = max(self.test_num_updates, num_updates)
            outputbs = [[]]*num_updates
            lossesb = [[]]*num_updates
            accuraciesb = [[]]*num_updates

            def task_metalearn(inp, reuse=True):
                """ Perform gradient descent for one task in the meta-batch. """
                inputa, inputb, labela, labelb = inp
                task_outputbs, task_lossesb = [], []

                task_outputa = self.forward(inputa, weights, reuse=reuse)  # only reuse on the first iter
                task_lossa = self.loss_func(task_outputa, labela)

                grads = tf.gradients(task_lossa, list(weights.values()))
                gradients = dict(zip(weights.keys(), grads))
                fast_weights = dict(zip(weights.keys(), [weights[key] - self.update_lr*gradients[key] for key in weights.keys()]))
                output = self.forward(inputb, fast_weights, reuse=True)
                task_outputbs.append(output)
                task_lossesb.append(self.loss_func(output, labelb))

                for j in range(num_updates - 1):
                    loss = self.loss_func(self.forward(inputa, fast_weights, reuse=True), labela)
                    grads = tf.gradients(loss, list(fast_weights.values()))
                    gradients = dict(zip(fast_weights.keys(), grads))
                    fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - self.update_lr*gradients[key]
                                                                  for key in fast_weights.keys()]))
                    output = self.forward(inputb, fast_weights, reuse=True)
                    task_outputbs.append(output)
                    task_lossesb.append(self.loss_func(output, labelb))

                task_output = [task_outputa, task_outputbs, task_lossa, task_lossesb]

                return task_output

            out_dtype = [tf.float32, [tf.float32]*num_updates, tf.float32, [tf.float32]*num_updates]
            result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb), dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)
            outputas, outputbs, lossesa, lossesb  = result

        ## Performance & Optimization
        if 'train' in prefix:
            self.total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(meta_batch_size)
            self.total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(meta_batch_size) for j in range(num_updates)]
            # after the map_fn
            self.outputas, self.outputbs = outputas, outputbs
            self.pretrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(total_loss1)

            if metatrain_iterations > 0:
                optimizer = tf.train.AdamOptimizer(self.meta_lr)
                self.gvs = gvs = optimizer.compute_gradients(self.total_losses2[num_updates-1])
                self.metatrain_op = optimizer.apply_gradients(gvs)
        else:
            self.metaval_total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(meta_batch_size)
            self.metaval_total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(meta_batch_size) for j in range(num_updates)]

        ## Summaries
        tf.summary.scalar(prefix+'Pre-update loss', total_loss1)

        for j in range(num_updates):
            tf.summary.scalar(prefix+'Post-update loss, step ' + str(j+1), total_losses2[j])

    def train(model, saver, sess, exp_string, data_generator, resume_itr=0,
              update_batch_size=20,
              num_updates=10,
              pretrain_iterations=1000,
              metatrain_iterations=100
              ):

        SUMMARY_INTERVAL = 100
        SAVE_INTERVAL = 1000
        PRINT_INTERVAL = 1000
        TEST_PRINT_INTERVAL = PRINT_INTERVAL * 5

        print('Done initializing, starting training.')
        prelosses, postlosses = [], []

        num_classes = data_generator.num_classes  # for classification, 1 otherwise
        multitask_weights, reg_weights = [], []

        for itr in range(resume_itr, pretrain_iterations + metatrain_iterations):
            feed_dict = {}
            if 'generate' in dir(data_generator):
                batch_x, batch_y, amp, phase = data_generator.generate()

                inputa = batch_x[:, :num_classes * update_batch_size, :]
                labela = batch_y[:, :num_classes * update_batch_size, :]
                inputb = batch_x[:, num_classes * update_batch_size:, :]  # b used for testing
                labelb = batch_y[:, num_classes * update_batch_size:, :]
                feed_dict = {model.inputa: inputa, model.inputb: inputb, model.labela: labela, model.labelb: labelb}

            if itr < pretrain_iterations:
                input_tensors = [model.pretrain_op]
            else:
                input_tensors = [model.metatrain_op]

            if (itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0):
                input_tensors.extend([model.summ_op, model.total_loss1, model.total_losses2[num_updates - 1]])

            result = sess.run(input_tensors, feed_dict)

            if itr % SUMMARY_INTERVAL == 0:
                prelosses.append(result[-2])
                postlosses.append(result[-1])

            if (itr != 0) and itr % PRINT_INTERVAL == 0:
                if itr < pretrain_iterations:
                    print_str = 'Pretrain Iteration ' + str(itr)
                else:
                    print_str = 'Iteration ' + str(itr - pretrain_iterations)
                print_str += ': ' + str(np.mean(prelosses)) + ', ' + str(np.mean(postlosses))
                print(print_str)
                prelosses, postlosses = [], []


            # sinusoid is infinite data, so no need to test on meta-validation set.
            if (itr != 0) and itr % TEST_PRINT_INTERVAL == 0:
                if 'generate' not in dir(data_generator):
                    feed_dict = {}
                    input_tensors = [model.metaval_total_loss1,
                                     model.metaval_total_losses2[num_updates - 1],
                                         model.summ_op]
                else:
                    batch_x, batch_y, amp, phase = data_generator.generate(train=False)
                    inputa = batch_x[:, :num_classes * update_batch_size, :]
                    inputb = batch_x[:, num_classes * update_batch_size:, :]
                    labela = batch_y[:, :num_classes * update_batch_size, :]
                    labelb = batch_y[:, num_classes * update_batch_size:, :]
                    feed_dict = {model.inputa: inputa, model.inputb: inputb, model.labela: labela, model.labelb: labelb,
                                 model.meta_lr: 0.0}
                    input_tensors = [model.total_loss1, model.total_losses2[num_updates - 1]]

                result = sess.run(input_tensors, feed_dict)
                print('Validation results: ' + str(result[0]) + ', ' + str(result[1]))

    ### Network construction functions (fc networks and conv networks)
    def construct_fc_weights(self):
        weights = {}
        weights['w1'] = tf.Variable(tf.truncated_normal([self.dim_input, self.dim_hidden[0]], stddev=0.01))
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden[0]]))
        for i in range(1,len(self.dim_hidden)):
            weights['w'+str(i+1)] = tf.Variable(tf.truncated_normal([self.dim_hidden[i-1], self.dim_hidden[i]], stddev=0.01))
            weights['b'+str(i+1)] = tf.Variable(tf.zeros([self.dim_hidden[i]]))
        weights['w'+str(len(self.dim_hidden)+1)] = tf.Variable(tf.truncated_normal([self.dim_hidden[-1], self.dim_output], stddev=0.01))
        weights['b'+str(len(self.dim_hidden)+1)] = tf.Variable(tf.zeros([self.dim_output]))
        return weights

    def forward_fc(self, inp, weights, reuse=False):
        hidden = normalize(tf.matmul(inp, weights['w1']) + weights['b1'], activation=tf.nn.relu, reuse=reuse, scope='0')
        for i in range(1,len(self.dim_hidden)):
            hidden = normalize(tf.matmul(hidden, weights['w'+str(i+1)]) + weights['b'+str(i+1)], activation=tf.nn.relu, reuse=reuse, scope=str(i+1))
        return tf.matmul(hidden, weights['w'+str(len(self.dim_hidden)+1)]) + weights['b'+str(len(self.dim_hidden)+1)]
