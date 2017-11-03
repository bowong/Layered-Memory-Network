import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import numpy as np

def get_fans(shape):
	if len(shape) == 2:
		fan_in = shape[0]
		fan_out = shape[1]
	elif len(shape) == 4 or len(shape) == 5:
		receptive_field_size = np.prod(shape[2:])
		fan_in = shape[1] * receptive_field_size
		fan_out = shape[0] * receptive_field_size

	else:
		# No specific assumptions.
		fan_in = np.sqrt(np.prod(shape))
		fan_out = np.sqrt(np.prod(shape))
	return fan_in, fan_out


def uniform(shape, scale=0.05, name=None, seed=None): #tf.float32
	if seed is None:
		# ensure that randomness is conditioned by the Numpy RNG
		seed = np.random.randint(10e8)

	value = tf.random_uniform_initializer(
		-scale, scale, dtype=tf.float32, seed=seed)(shape)

	return tf.Variable(value,name=name)
    


def glorot_uniform(shape, name=None):
    fan_in, fan_out = get_fans(shape)
    s = np.sqrt(6. / (fan_in + fan_out))
    return uniform(shape, s, name=name)


def orthogonal(shape, scale=1.1, name=None):
    """Orthogonal initializer.

    # References
        Saxe et al., http://arxiv.org/abs/1312.6120
    """
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # Pick the one with the correct shape.
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return tf.Variable(scale * q[:shape[0], :shape[1]], dtype=tf.float32, name=name)

def init_weight_variable(shape, init_method='glorot_uniform', name=None):
	# initial = tf.truncated_normal(shape, stddev=0.1, name=name)
	if init_method == 'uniform':
		return uniform(shape, scale=0.05, name=name, seed=None)
	elif init_method == 'glorot_uniform':
		return glorot_uniform(shape, name=name)
	elif init_method == 'orthogonal':
		return orthogonal(shape, scale=1.1, name=name)
	else:
		raise ValueError('Invalid init_method: ' + init_method)
	
def init_bias_variable(shape,name=None):
	initial = tf.constant(0.1,shape=shape, name=name)
	return tf.Variable(initial, name=name)