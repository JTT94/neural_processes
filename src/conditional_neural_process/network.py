import tensorflow as tf


### Network construction functions (fc networks and conv networks)
def construct_fc_weights(dim_input, dim_hidden, dim_output):
  weights = {}
  weights['w1'] = tf.Variable(tf.truncated_normal([dim_input, dim_hidden[0]], stddev=0.01))
  weights['b1'] = tf.Variable(tf.zeros([dim_hidden[0]]))
  for i in range(1, len(dim_hidden)):
    weights['w' + str(i + 1)] = tf.Variable(
      tf.truncated_normal([dim_hidden[i - 1], dim_hidden[i]], stddev=0.01))
    weights['b' + str(i + 1)] = tf.Variable(tf.zeros([dim_hidden[i]]))
  weights['w' + str(len(dim_hidden) + 1)] = tf.Variable(
    tf.truncated_normal([dim_hidden[-1], dim_output], stddev=0.01))
  weights['b' + str(len(dim_hidden) + 1)] = tf.Variable(tf.zeros([dim_output]))
  return weights


def forward_fc(inp, weights, dim_hidden, reuse=False):
  hidden = tf.nn.relu(tf.matmul(inp, weights['w1']) + weights['b1'])
  for i in range(1, len(dim_hidden)):
    hidden = tf.nn.relu(tf.matmul(hidden, weights['w' + str(i + 1)]) + weights['b' + str(i + 1)])
  return tf.matmul(hidden, weights['w' + str(len(dim_hidden) + 1)]) + weights['b' + str(len(dim_hidden) + 1)]

class DeterministicEncoder(object):
  """The Encoder."""

  def __init__(self, output_sizes):
    """CNP encoder.

    Args:
      output_sizes: An iterable containing the output sizes of the encoding MLP.
    """
    self._output_sizes = output_sizes

  def __call__(self, context_x, context_y, num_context_points):
    """Encodes the inputs into one representation.

    Args:
      context_x: Tensor of size bs x observations x m_ch. For this 1D regression
          task this corresponds to the x-values.
      context_y: Tensor of size bs x observations x d_ch. For this 1D regression
          task this corresponds to the y-values.
      num_context_points: A tensor containing a single scalar that indicates the
          number of context_points provided in this iteration.

    Returns:
      representation: The encoded representation averaged over all context
          points.
    """

    # Concatenate x and y along the filter axes
    encoder_input = tf.concat([context_x, context_y], axis=-1)

    # Get the shapes of the input and reshape to parallelise across observations
    batch_size, _, filter_size = encoder_input.shape.as_list()
    hidden = tf.reshape(encoder_input, (batch_size * num_context_points, -1))
    hidden.set_shape((None, filter_size))

    dim_input = filter_size
    dim_hidden = self._output_sizes[:-1]
    dim_output = self._output_sizes[-1]
    # Pass through MLP
    with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
      self.weights = construct_fc_weights(dim_input, dim_hidden, dim_output)
      hidden = forward_fc(hidden, self.weights, dim_hidden, reuse=False)

    size = self._output_sizes[-2]
    # Bring back into original shape
    hidden = tf.reshape(hidden, (batch_size, num_context_points, size))

    # Aggregator: take the mean over all points
    representation = tf.reduce_mean(hidden, axis=1)

    return representation

  class DeterministicDecoder(object):
    """The Decoder."""

    def __init__(self, output_sizes):
      """CNP decoder.

      Args:
        output_sizes: An iterable containing the output sizes of the decoder MLP
            as defined in `basic.Linear`.
      """
      self._output_sizes = output_sizes

    def __call__(self, representation, target_x, num_total_points):
      """Decodes the individual targets.

      Args:
        representation: The encoded representation of the context
        target_x: The x locations for the target query
        num_total_points: The number of target points.

      Returns:
        dist: A multivariate Gaussian over the target points.
        mu: The mean of the multivariate Gaussian.
        sigma: The standard deviation of the multivariate Gaussian.
      """

      # Concatenate the representation and the target_x
      representation = tf.tile(
        tf.expand_dims(representation, axis=1), [1, num_total_points, 1])
      input = tf.concat([representation, target_x], axis=-1)

      # Get the shapes of the input and reshape to parallelise across observations
      batch_size, _, filter_size = input.shape.as_list()
      hidden = tf.reshape(input, (batch_size * num_total_points, -1))
      hidden.set_shape((None, filter_size))

      dim_input = filter_size
      dim_hidden = self._output_sizes[:-1]
      dim_output = self._output_sizes[-1]

      # Pass through MLP
      with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
        self.weights = construct_fc_weights(dim_input, dim_hidden, dim_output)
        hidden = forward_fc(hidden, self.weights, dim_hidden, reuse=False)

      # Bring back into original shape
      hidden = tf.reshape(hidden, (batch_size, num_total_points, -1))

      # Get the mean an the variance
      mu, log_sigma = tf.split(hidden, 2, axis=-1)

      # Bound the variance
      sigma = 0.1 + 0.9 * tf.nn.softplus(log_sigma)

      # Get the distribution
      dist = tf.contrib.distributions.MultivariateNormalDiag(
        loc=mu, scale_diag=sigma)

      return dist, mu, sigma