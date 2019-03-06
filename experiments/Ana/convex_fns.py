from src.neuralprocesses import NeuralProcessParams, split_context_target
from src.neuralprocesses.network import encoder_h, decoder_g, xy_to_z_params
from src.neuralprocesses.process import init_neural_process
from src.neuralprocesses.predict import posterior_predict
import numpy as np
import random
import matplotlib.pyplot as plt

import tensorflow as tf


#=========================================================
# INITIALISATION

# Initial setup
params = NeuralProcessParams(dim_r=2, dim_z=2, n_hidden_units_h=[64, 64], n_hidden_units_g=[64, 64, 64],
                             noise_std=0.05)

tf.reset_default_graph()
sess = tf.Session()

# Placeholders for training inputs
x_context = tf.placeholder(tf.float32, (None, 1))
y_context = tf.placeholder(tf.float32, (None, 1))
x_target = tf.placeholder(tf.float32, (None, 1))
y_target = tf.placeholder(tf.float32, (None, 1))

# Set up NN
train_op, loss = init_neural_process(x_context, y_context, x_target, y_target,
                                     params, encoder_h, decoder_g, learning_rate=0.001)

# Initialise
init = tf.global_variables_initializer()
sess.run(init)

n_iter = 1000
loss_freq = n_iter/10
n_obs = 100
q_low = 5
q_high = 10
p_low = -2
p_high = 2
x_range_low = -5
x_range_high = 5
q_pred = 3
p_pred = 0

#=========================================================
# TRAINING

train_xs = []
train_q = []
train_ys = []
train_p = []

for i in range(n_iter):
    xs = np.random.uniform(x_range_low, x_range_high, n_obs)
    q = random.uniform(q_low, q_high)
    p = random.uniform(p_low, p_high)
    ys = q * (xs)**2 + p

    train_xs.append(xs)
    train_q.append(q)
    train_p.append(p)
    train_ys.append(ys)

    n_context = random.choice(range(1, 99))
    feed_dict = split_context_target(xs.reshape(-1, 1), ys.reshape(-1, 1), n_context, x_context, y_context, x_target,
                                     y_target)
    a = sess.run((train_op, loss), feed_dict=feed_dict)
    if i % loss_freq == 0:
        print("Loss: {:.3f}".format(a[1]))


#=========================================================
# PLOTTING PREDICTIONS

xs = np.random.uniform(x_range_low, x_range_high, 5)
ys = q_pred * (xs)**2 + p_pred
x_star = np.linspace(x_range_low, x_range_high, 100)
y_star = q_pred * (x_star)**2 + p_pred

def plot_prediction(ax, xs, ys, x_star, y_star, plot_true = True, xlim = (-4.5, 4.5), ylim=(-1.5, 1.5), sess= tf.get_default_session()):
    posterior_predict_op = posterior_predict(
        xs.reshape((-1,1)),
        ys.reshape((-1,1)),
        x_star.reshape((-1,1)),
        params, encoder_h, decoder_g, n_draws=50)
    y_star_mat = sess.run(posterior_predict_op.mu)

    for i in range(y_star_mat.shape[1]):
        ax.plot(x_star, y_star_mat.T[i], c='b', alpha=0.1)
    if plot_true:
        ax.plot(x_star, y_star, c='r', alpha=.5)
    ax.scatter(xs, ys, c='r')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

fig, axes = plt.subplots(3, 1, figsize=(5,5))
xss = [np.array(xs) for xs in [[0], [-1, 1], [-2, -1, 0, 1, 2]]]
yss = [q_pred * (xs)**2 + p_pred for xs in xss]
plot_true = False
ylim=(-30, 30)
for ax, xs, ys in zip(axes, xss, yss):
    plot_prediction(ax, xs, ys, x_star, y_star, plot_true = plot_true, ylim = ylim, sess=sess)
    plot_true = True
    ylim=(-30, 30)

plt.show()






