import tensorflow as tf
from src.conditional_neural_process.gp_data_generator import GPCurvesReader
from src.conditional_neural_process.model import DeterministicModel

# Config
#------------------------------------------------------------
TRAINING_ITERATIONS = int(100)
MAX_CONTEXT_POINTS = 10
PLOT_AFTER = int(2e4)
tf.reset_default_graph()

# Data
#------------------------------------------------------------
# Train dataset
dataset_train = GPCurvesReader(
    batch_size=64, max_num_context=MAX_CONTEXT_POINTS)
data_train = dataset_train.generate_curves()

# Test dataset
dataset_test = GPCurvesReader(
    batch_size=1, max_num_context=MAX_CONTEXT_POINTS, testing=True)
data_test = dataset_test.generate_curves()


dataset_train_task_level = GPCurvesReader(
    batch_size=1, max_num_context=MAX_CONTEXT_POINTS)
data_task_train = dataset_train.generate_curves()

# Test dataset
dataset_test = GPCurvesReader(
    batch_size=1, max_num_context=MAX_CONTEXT_POINTS, testing=True)
data_test = dataset_test.generate_curves()

# Model
#------------------------------------------------------------
# Sizes of the layers of the MLPs for the encoder and decoder
# The final output layer of the decoder outputs two values, one for the mean and
# one for the variance of the prediction at the target location
encoder_output_sizes = [128, 128, 128, 128]
decoder_output_sizes = [128, 128, 2]

# Define the model
def loss_fn(context_x, context_y, target_x, target_y, encoder_weights, decoder_weights):
  # Pass query through the encoder and the decoder
  representation = model.encoder(context_x, context_y, data_task_train.num_context_points, weights=encoder_weights)
  dist, mu, sigma = model.decoder(representation, target_x, data_task_train.num_total_points, weights=decoder_weights)

  log_p = dist.log_prob(target_y)
  loss = -tf.reduce_mean(log_p)
  return loss

def update_weights(loss, weights, update_lr = 0.2):
  grads = tf.gradients(loss, list(weights.values()))
  gradients = dict(zip(weights.keys(), grads))
  fast_weights = dict(zip(weights.keys(), [weights[key] - update_lr*gradients[key] for key in weights.keys()]))
  return fast_weights

model = DeterministicModel(encoder_output_sizes, decoder_output_sizes)
# Hack to initialise weights
_, _, _ = model(data_train.query, data_train.num_total_points,
                       data_train.num_context_points, data_train.target_y)

lossesb = []
batch_size = 10
encoder_weights = model.encoder.weights
decoder_weights = model.decoder.weights
for i in range(batch_size):
  (context_x, context_y), target_x = data_task_train.query
  target_y = data_task_train.target_y
  lossa = loss_fn(context_x, context_y, target_x, target_y, encoder_weights, decoder_weights)
  new_encoder_weights = update_weights(lossa, encoder_weights, update_lr = 0.2)
  new_decoder_weights = update_weights(lossa, decoder_weights, update_lr = 0.2)

  (context_xb, context_yb), target_xb = data_task_train.query
  target_yb = data_task_train.target_y
  lossb = loss_fn(context_xb, context_yb, target_xb, target_yb, new_encoder_weights, new_decoder_weights)
  lossesb.append(lossb)

total_loss_b = tf.reduce_sum(lossesb)
optimizer = tf.train.AdamOptimizer(0.2)
gvs = optimizer.compute_gradients(total_loss_b)
metatrain_op = optimizer.apply_gradients(gvs)


# Training loop
#------------------------------------------------------------

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
for it in range(TRAINING_ITERATIONS):
  sess.run([metatrain_op])

