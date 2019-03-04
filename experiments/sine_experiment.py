from src.neuralprocesses import NeuralProcessParams, split_context_target
from src.neuralprocesses.network import encoder_h, decoder_g, xy_to_z_params
from src.neuralprocesses.process import init_neural_process
from src.neuralprocesses.predict import posterior_predict
import numpy as np
import random
import matplotlib.pyplot as plt

import tensorflow as tf

params = NeuralProcessParams(dim_r=2, dim_z=2, n_hidden_units_h=[8], n_hidden_units_g=[32, 32, 32])

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

n_iter = 100000
loss_freq = n_iter/10

train_xs = []
train_amp = []
train_ys = []
train_phase = []
for i in range(n_iter):
    n_obs = 20
    xs = np.random.uniform(-3, 3, n_obs)
    amp = random.uniform(-2, 2)
    phase = random.uniform(0, np.pi/2)
    ys = amp * np.sin(xs + phase)

    train_xs.append(xs)
    train_amp.append(amp)
    train_ys.append(ys)
    train_phase.append(phase)

    n_context = random.choice(range(1, 11))
    feed_dict = split_context_target(xs.reshape(-1, 1), ys.reshape(-1, 1), n_context, x_context, y_context, x_target,
                                     y_target)
    a = sess.run((train_op, loss), feed_dict=feed_dict)
    if i % loss_freq == 0:
        print("Loss: {:.3f}".format(a[1]))


amp = 0.5
phase = np.pi/4
xs = np.random.uniform(-3, 3, 5)
ys = amp * np.sin(xs + phase)
x_star = np.linspace(-3, 3, 100)
y_star = amp * np.sin(x_star+phase)

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

fig, axes = plt.subplots(3, 1, figsize=(16,16))
xss = [np.array(xs) for xs in [[0], [0, 1], [-2, -1, 0, 1, 2]]]
yss = [amp*np.sin(xs+phase) for xs in xss]
plot_true = False
ylim=(-0.5, 0.5)
for ax, xs, ys in zip(axes, xss, yss):
    plot_prediction(ax, xs, ys, x_star, y_star, plot_true = plot_true, ylim = ylim, sess=sess)
    plot_true = True
    ylim=(-1.5, 1.5)

plt.show()