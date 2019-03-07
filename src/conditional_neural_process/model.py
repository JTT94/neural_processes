import tensorflow as tf
from .network import DeterministicEncoder, DeterministicDecoder

class DeterministicModel(object):
  """The CNP model."""

  def __init__(self, encoder_output_sizes, decoder_output_sizes):
    """Initialises the model.

    Args:
      encoder_output_sizes: An iterable containing the sizes of hidden layers of
          the encoder. The last one is the size of the representation r.
      decoder_output_sizes: An iterable containing the sizes of hidden layers of
          the decoder. The last element should correspond to the dimension of
          the y * 2 (it encodes both mean and variance concatenated)
    """
    self.encoder = DeterministicEncoder(encoder_output_sizes)
    self.decoder = DeterministicDecoder(decoder_output_sizes)

  def __call__(self, query, num_total_points, num_contexts, target_y=None):
    """Returns the predicted mean and variance at the target points.

    Args:
      query: Array containing ((context_x, context_y), target_x) where:
          context_x: Array of shape batch_size x num_context x 1 contains the
              x values of the context points.
          context_y: Array of shape batch_size x num_context x 1 contains the
              y values of the context points.
          target_x: Array of shape batch_size x num_target x 1 contains the
              x values of the target points.
      target_y: The ground truth y values of the target y. An array of
          shape batchsize x num_targets x 1.
      num_total_points: Number of target points.

    Returns:
      log_p: The log_probability of the target_y given the predicted
      distribution.
      mu: The mean of the predicted distribution.
      sigma: The variance of the predicted distribution.
    """

    (context_x, context_y), target_x = query

    # Pass query through the encoder and the decoder
    representation = self.encoder(context_x, context_y, num_contexts)
    dist, mu, sigma = self.decoder(representation, target_x, num_total_points)

    # If we want to calculate the log_prob for training we will make use of the
    # target_y. At test time the target_y is not available so we return None
    if target_y is not None:
      log_p = dist.log_prob(target_y)
    else:
      log_p = None

    return log_p, mu, sigma