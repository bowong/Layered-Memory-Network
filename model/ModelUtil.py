import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf

import numpy as np
from sklearn.decomposition import PCA
import cPickle as pickle

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

	return tf.Variable(value)
    


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
	return tf.Variable(initial)


def matmul_wx(x, w, b, output_dims):
	
	return tf.matmul(x, w)+tf.reshape(b,(1,output_dims))
	

def matmul_uh(u,h_tm1):
	return tf.matmul(h_tm1,u)



def get_init_state(x, output_dims):
	initial_state = tf.zeros_like(x)
	initial_state = tf.reduce_sum(initial_state,axis=[1,2])
	initial_state = tf.expand_dims(initial_state,dim=-1)
	initial_state = tf.tile(initial_state,[1,output_dims])
	return initial_state


def getVideoEncoder(x, output_dims, return_sequences=False):
	'''
		function: getVideoEncoder
		parameters:

			x: batch_size, timesteps , dims
			output_dims: the output of the GRU dimensions
			num_class: number of class : ucf-101: 101
		return:
			the last GRU state, 
			or
			the sequences of the hidden states

	'''
	input_shape = x.get_shape().as_list()
	assert len(input_shape)==3 
	timesteps = input_shape[1]
	input_dims = input_shape[2]

	# get initial state
	initial_state = get_init_state(x, output_dims)

	# initialize the parameters
	# W_r,U_r,b_r; W_z, U_z, b_z; W_h, U_h, b_h
	W_r = init_weight_variable((input_dims,output_dims),init_method='glorot_uniform',name="W_r")
	W_z = init_weight_variable((input_dims,output_dims),init_method='glorot_uniform',name="W_z")
	W_h = init_weight_variable((input_dims,output_dims),init_method='glorot_uniform',name="W_h")

	U_r = init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_r")
	U_z = init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_z")
	U_h = init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_h")

	b_r = init_bias_variable((output_dims,),name="b_r")
	b_z = init_bias_variable((output_dims,),name="b_z")
	b_h = init_bias_variable((output_dims,),name="b_h")


	# batch_size x timesteps x dim -> timesteps x batch_size x dim
	axis = [1,0]+list(range(2,3))  # axis = [1,0,2]
	x = tf.transpose(x, perm=axis) # permutate the input_x --> timestemp, batch_size, input_dims

	input_x = tf.TensorArray(
            dtype=x.dtype,
            size=timesteps,
            tensor_array_name='input_x')

	if hasattr(input_x, 'unstack'):
		input_x = input_x.unstack(x)
	else:
		input_x = input_x.unpack(x)	


	hidden_state = tf.TensorArray(
            dtype=tf.float32,
            size=timesteps,
            tensor_array_name='hidden_state')




	def step(time, hidden_state, h_tm1):
		x_t = input_x.read(time) # batch_size * dim

		preprocess_x_r = matmul_wx(x_t, W_r, b_r, output_dims)
		preprocess_x_z = matmul_wx(x_t, W_z, b_z, output_dims)
		preprocess_x_h = matmul_wx(x_t, W_h, b_h, output_dims)

		r = tf.nn.sigmoid(preprocess_x_r+ matmul_uh(U_r,h_tm1))
		z = tf.nn.sigmoid(preprocess_x_z+ matmul_uh(U_z,h_tm1))
		hh = tf.nn.tanh(preprocess_x_h+ matmul_uh(U_h,h_tm1))

		h = (1-z)*hh + z*h_tm1

		hidden_state = hidden_state.write(time, h)

		return (time+1,hidden_state,h)

	


	time = tf.constant(0, dtype='int32', name='time')


	ret_out = tf.while_loop(
            cond=lambda time, *_: time < timesteps,
            body=step,
            loop_vars=(time, hidden_state, initial_state),
            parallel_iterations=32,
            swap_memory=True)

	output = ret_out[1]
	last_output = ret_out[-1] 

	if hasattr(hidden_state, 'stack'):
		hidden_state = hidden_state.stack()

	axis = [1,0] + list(range(2,3))
	outputs = tf.transpose(hidden_state,perm=axis)


	if return_sequences:
		return outputs
	else:
		return last_output


def getEmbedding(words, size_voc, word_embedding_size):
	'''
		function: getEmbedding
		parameters:
			words: int, word index ; or a np.int32 list ## sample(null) * input_words_sequential
			size_voc: size of vocabulary
			embedding_size: the dimension after embedding
		return:
			embeded_words:the embeded words with shape (sample * timesteps * embedding dims)
			mask: each element in mask vector is 0 or 1,  indicate there is a word or a padding zero
	'''

	W_e = tf.get_variable('W_e',(size_voc,word_embedding_size),initializer=tf.random_uniform_initializer(-0.05,0.05)) # share the embedding matrix
	embeded_words = tf.gather(W_e, words)
	mask =  tf.not_equal(words,0)
	return embeded_words, mask 



def getQuestionEncoder(embeded_words, output_dims, mask, return_sequences=False):

	'''
		function: getQuestionEncoder
		parameters:
			embeded_words: sample*timestep*dim
			output_dims: the GRU hidden dim
			mask: bool type , samples * timestep
		return:
			the last GRU state, 
			or
			the sequences of the hidden states
	'''
	input_shape = embeded_words.get_shape().as_list()
	assert len(input_shape)==3 

	timesteps = input_shape[1]
	input_dims = input_shape[2]
	# get initial state
	initial_state = get_init_state(embeded_words, output_dims)


	# initialize the parameters
	# W_r,U_r,b_r; W_z, U_z, b_z; W_h, U_h, b_h
	W_r = init_weight_variable((input_dims,output_dims),init_method='glorot_uniform',name="W_q_r")
	W_z = init_weight_variable((input_dims,output_dims),init_method='glorot_uniform',name="W_q_z")
	W_h = init_weight_variable((input_dims,output_dims),init_method='glorot_uniform',name="W_q_h")

	U_r = init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_q_r")
	U_z = init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_q_z")
	U_h = init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_q_h")

	b_r = init_bias_variable((output_dims,),name="b_q_r")
	b_z = init_bias_variable((output_dims,),name="b_q_z")
	b_h = init_bias_variable((output_dims,),name="b_q_h")


	# batch_size x timesteps x dim -> timesteps x batch_size x dim
	axis = [1,0]+list(range(2,3))  # axis = [1,0,2]
	embeded_words = tf.transpose(embeded_words, perm=axis) # permutate the input_x --> timestemp, batch_size, input_dims



	input_embeded_words = tf.TensorArray(
            dtype=embeded_words.dtype,
            size=timesteps,
            tensor_array_name='input_embeded_words_q')


	if hasattr(input_embeded_words, 'unstack'):
		input_embeded_words = input_embeded_words.unstack(embeded_words)
	else:
		input_embeded_words = input_embeded_words.unpack(embeded_words)	


	# preprocess mask
	if len(mask.get_shape()) == len(input_shape)-1:
		mask = tf.expand_dims(mask,dim=-1)
	
	mask = tf.transpose(mask,perm=axis)

	input_mask = tf.TensorArray(
		dtype=mask.dtype,
		size=timesteps,
		tensor_array_name='input_mask_q'
		)

	if hasattr(input_mask, 'unstack'):
		input_mask = input_mask.unstack(mask)
	else:
		input_mask = input_mask.unpack(mask)


	hidden_state_q = tf.TensorArray(
            dtype=tf.float32,
            size=timesteps,
            tensor_array_name='hidden_state_q')



	def step(time, hidden_state_q, h_tm1):
		x_t = input_embeded_words.read(time) # batch_size * dim
		mask_t = input_mask.read(time)

		preprocess_x_r = matmul_wx(x_t, W_r, b_r, output_dims)
		preprocess_x_z = matmul_wx(x_t, W_z, b_z, output_dims)
		preprocess_x_h = matmul_wx(x_t, W_h, b_h, output_dims)

		r = tf.nn.sigmoid(preprocess_x_r+ matmul_uh(U_r,h_tm1))
		z = tf.nn.sigmoid(preprocess_x_z+ matmul_uh(U_z,h_tm1))
		hh = tf.nn.tanh(preprocess_x_h+ matmul_uh(U_h,h_tm1))

		
		h = (1-z)*hh + z*h_tm1
		tiled_mask_t = tf.tile(mask_t, tf.stack([1, h.get_shape().as_list()[1]]))

		h = tf.where(tiled_mask_t, h, h_tm1)
		
		hidden_state_q = hidden_state_q.write(time, h)

		return (time+1,hidden_state_q,h)

	


	time = tf.constant(0, dtype='int32', name='time')


	ret_out = tf.while_loop(
            cond=lambda time, *_: time < timesteps,
            body=step,
            loop_vars=(time, hidden_state_q, initial_state),
            parallel_iterations=32,
            swap_memory=True)


	hidden_state_q = ret_out[1]
	last_output = ret_out[-1] 
	
	if hasattr(hidden_state_q, 'stack'):
		outputs = hidden_state_q.stack()
		print('stack')
	else:
		outputs = hidden_state_q.pack()

	axis = [1,0] + list(range(2,3))
	outputs = tf.transpose(outputs,perm=axis)

	if return_sequences:
		return outputs
	else:
		return last_output




def getAnswerEmbedding(words, size_voc, word_embedding_size):
	'''
		function: getAnswerEmbedding
		parameters:
			words: int, word index ; or a np.int32 list ## sample(null) * numebrOfChoice * timesteps
			size_voc: size of vocabulary
			embedding_size: the dimension after embedding
		return:
			the embeded answers with shape(batch_size, numberOfChoices, timesteps, word_embedding_size)
	'''
	assert len(words.get_shape().as_list())==3 #
	input_shape = words.get_shape().as_list()
	numberOfChoices = input_shape[1]
	timesteps = input_shape[2]

	mask =  tf.not_equal(words,0)

	words = tf.reshape(words, (-1,timesteps))
	W_e = tf.get_variable('W_e',(size_voc,word_embedding_size),initializer=tf.random_uniform_initializer(-0.05,0.05)) # share the embedding matrix
	embeded_words = tf.gather(W_e, words)
	

	embeded_words = tf.reshape(embeded_words,(-1,numberOfChoices,timesteps,word_embedding_size))
	
	return embeded_words, mask 



def getAnswerEncoder(embeded_words, output_dims, mask, return_sequences=False):
	'''
		function: getAnswerEncoder
		parameters:
			embeded_words: samples * numberOfChoices * timesteps  * dim
			output_dim: output of GRU, the dimension of answering vector
			mask : bool type, mask the embeded_words
			num_class: number of classifier
		return:
			the last encoded answers with shape(batch_size, numberOfChoices, output_dims)
			or
			the sequences.... with shape(batch_size, numberOfChoices, numberOfChoices, output_dims)
	'''
	input_shape = embeded_words.get_shape().as_list()
	assert len(input_shape)==4 


	numberOfChoices = input_shape[1]
	timesteps = input_shape[2]
	input_dims = input_shape[3]

	# get initial state
	embeded_words = tf.reshape(embeded_words,(-1,timesteps,input_dims))
	initial_state = get_init_state(embeded_words, output_dims)

	axis = [1,0,2]  
	embeded_words = tf.transpose(embeded_words, perm=axis) # permutate the 'embeded_words' --> timesteps x batch_size x numberOfChoices x dim
	# embeded_words = tf.reshape(embeded_words,(timesteps,-1,input_dims)) # reshape the 'embeded_words' --> timesteps x (batch x numberOfChoices) x dim
	
	# initialize the parameters
	# W_r,U_r,b_r; W_z, U_z, b_z; W_h, U_h, b_h
	W_r = init_weight_variable((input_dims,output_dims),init_method='glorot_uniform',name="W_a_r")
	W_z = init_weight_variable((input_dims,output_dims),init_method='glorot_uniform',name="W_a_z")
	W_h = init_weight_variable((input_dims,output_dims),init_method='glorot_uniform',name="W_a_h")

	U_r = init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_a_r")
	U_z = init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_a_z")
	U_h = init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_a_h")

	b_r = init_bias_variable((output_dims,),name="b_a_r")
	b_z = init_bias_variable((output_dims,),name="b_a_z")
	b_h = init_bias_variable((output_dims,),name="b_a_h")



	input_embeded_words = tf.TensorArray(
            dtype=embeded_words.dtype,
            size=timesteps,
            tensor_array_name='input_embeded_words_a')


	if hasattr(input_embeded_words, 'unstack'):
		input_embeded_words = input_embeded_words.unstack(embeded_words)
	else:
		input_embeded_words = input_embeded_words.unpack(embeded_words)	


	# preprocess mask
	if len(mask.get_shape()) == len(input_shape)-1:
		mask = tf.expand_dims(mask,dim=-1)
	
	axis = [2,0,1,3]  
	mask = tf.transpose(mask,perm=axis)
	mask = tf.reshape(mask, (timesteps,-1,1))

	input_mask = tf.TensorArray(
		dtype=mask.dtype,
		size=timesteps,
		tensor_array_name='input_mask_q'
		)

	if hasattr(input_mask, 'unstack'):
		input_mask = input_mask.unstack(mask)
	else:
		input_mask = input_mask.unpack(mask)


	hidden_state_q = tf.TensorArray(
            dtype=tf.float32,
            size=timesteps,
            tensor_array_name='hidden_state_a')

	# if hasattr(hidden_state, 'unstack'):
	# 	hidden_state = hidden_state.unstack(hidden_state)
	# else:
	# 	hidden_state = hidden_state.unpack(hidden_state)


	def step(time, hidden_state_q, h_tm1):
		x_t = input_embeded_words.read(time) # batch_size * dim
		mask_t = input_mask.read(time)

		preprocess_x_r = matmul_wx(x_t, W_r, b_r, output_dims)
		preprocess_x_z = matmul_wx(x_t, W_z, b_z, output_dims)
		preprocess_x_h = matmul_wx(x_t, W_h, b_h, output_dims)

		r = tf.nn.sigmoid(preprocess_x_r+ matmul_uh(U_r,h_tm1))
		z = tf.nn.sigmoid(preprocess_x_z+ matmul_uh(U_z,h_tm1))
		hh = tf.nn.tanh(preprocess_x_h+ matmul_uh(U_h,h_tm1))

		
		h = (1-z)*hh + z*h_tm1
		tiled_mask_t = tf.tile(mask_t, tf.stack([1, h.get_shape().as_list()[1]]))

		h = tf.where(tiled_mask_t, h, h_tm1)
		
		hidden_state_q = hidden_state_q.write(time, h)

		return (time+1,hidden_state_q,h)

	


	time = tf.constant(0, dtype='int32', name='time')


	ret_out = tf.while_loop(
            cond=lambda time, *_: time < timesteps,
            body=step,
            loop_vars=(time, hidden_state_q, initial_state),
            parallel_iterations=32,
            swap_memory=True)


	hidden_state_q = ret_out[1]
	last_output = ret_out[-1] 


	
	if hasattr(hidden_state_q, 'stack'):
		outputs = hidden_state_q.stack()
		print('stack')
	else:
		outputs = hidden_state_q.pack()

	outputs = tf.reshape(outputs,(timesteps,-1,numberOfChoices,output_dims))
	axis = [1,2,0]+list(range(3,4))
	outputs = tf.transpose(outputs,perm=axis)

	last_output = tf.reshape(last_output,(-1,numberOfChoices,output_dims))
	print('outputs:....',outputs.get_shape().as_list())
	if return_sequences:
		return outputs
	else:
		return last_output




def getMemoryNetworks(embeded_stories, embeded_question, d_lproj, T_B=None, return_sequences=False):

	'''
		embeded_stories: (batch_size, num_of_sentence, num_of_words, embeded_words_dims)
		embeded_question:(batch_size, embeded_words_dims)
		output_dims: the dimension of stories 
	'''
	stories_shape = embeded_stories.get_shape().as_list()
	embeded_question_shape = embeded_question.get_shape().as_list()
	num_of_sentence = stories_shape[-3]
	input_dims = stories_shape[-1]
	output_dims = embeded_question_shape[-1]


	embeded_stories = getAverageRepresentation(embeded_stories, T_B, d_lproj)

	
	embeded_question = tf.tile(tf.expand_dims(embeded_question,dim=1),[1,num_of_sentence,1])

	sen_weight = tf.reduce_sum(embeded_question*embeded_stories,reduction_indices=-1,keep_dims=True)

	sen_weight = tf.nn.softmax(sen_weight,dim=1)
	sen_weight = tf.tile(sen_weight,[1,1,output_dims])
	if return_sequences:
		embeded_stories = embeded_stories*sen_weight
	else:
		embeded_stories = tf.reduce_sum(embeded_stories*sen_weight,reduction_indices=1) # (batch_size, output_dims)

	return embeded_stories

def getMemoryNetworksMaxPooling(embeded_stories, embeded_question, d_lproj, T_B=None):

	'''
		embeded_stories: (batch_size, num_of_sentence, num_of_words, embeded_words_dims)
		embeded_question:(batch_size, embeded_words_dims)
		output_dims: the dimension of stories 
	'''
	stories_shape = embeded_stories.get_shape().as_list()
	embeded_question_shape = embeded_question.get_shape().as_list()
	num_of_sentence = stories_shape[-3]
	input_dims = stories_shape[-1]
	output_dims = embeded_question_shape[-1]


	embeded_stories = getAverageRepresentation(embeded_stories, T_B, d_lproj)

	
	embeded_question = tf.tile(tf.expand_dims(embeded_question,dim=1),[1,num_of_sentence,1])

	sen_weight = tf.reduce_sum(embeded_question*embeded_stories,reduction_indices=-1,keep_dims=True)

	sen_weight = tf.nn.softmax(sen_weight,dim=1)
	sen_weight = tf.tile(sen_weight,[1,1,output_dims])

	embeded_stories = tf.reduce_max(embeded_stories*sen_weight,reduction_indices=1) # (batch_size, output_dims)

	return embeded_stories

rng = np.random
rng.seed(1234)

def init_linear_projection(rng, nrows, ncols, pca_mat=None):
    """ Linear projection (for example when using fixed w2v as LUT """
    if nrows == ncols:
        P = np.eye(nrows)
        print "Linear projection: initialized as identity matrix"
    else:
        assert([nrows, ncols] == pca_mat.shape, 'PCA matrix not of same size as RxC')
        P = 0.1 * pca_mat
        print "Linear projection: initialized with 0.1 PCA"

    return P.astype('float32')

def setWord2VecModelConfiguration(v2i,w2v,d_w2v,d_lproj):
	'''
		v2i: vocab(word) to int(index)
		w2v: word to vector
		d_w2v:dimension of w2v
		d_lproj: dimension of projection
	'''
	voc_size = len(v2i)
	np_mask = np.vstack((np.zeros(d_w2v),np.ones((voc_size-1,d_w2v))))
	T_mask = tf.constant(np_mask, tf.float32, name='LUT_mask')

	pca_mat = None
	print "Initialize LUTs as word2vec and use linear projection layer"


	LUT = np.zeros((voc_size, d_w2v), dtype='float32')
	found_words = 0

	for w, v in v2i.iteritems():
		if w in w2v.vocab:
			LUT[v] = w2v.get_vector(w)
			found_words +=1
		else:
			LUT[v] = rng.randn(d_w2v)
			LUT[v] = LUT[v] / (np.linalg.norm(LUT[v]) + 1e-6)

	print "Found %d / %d words" %(found_words, len(v2i))


	# word 0 is blanked out, word 1 is 'UNK'
	LUT[0] = np.zeros((d_w2v))

	# if linear projection layer is not the same shape as LUT, then initialize with PCA
	if d_lproj != LUT.shape[1]:
		pca = PCA(n_components=d_lproj, whiten=True)
		pca_mat = pca.fit_transform(LUT.T)  # 300 x 100?

	# setup LUT!
	T_w2v = tf.constant(LUT.astype('float32'))

	T_B = tf.Variable(init_linear_projection(rng, d_w2v, d_lproj, pca_mat), name='B')



	return T_B, T_w2v, T_mask, pca_mat


def getEmbeddingWithWord2Vec(words, T_w2v, T_mask):
	input_shape = words.get_shape().as_list()

	mask =  tf.not_equal(words,0)

	embeded_words = tf.gather(T_w2v,words)*tf.gather(T_mask,words)

	return embeded_words, mask 

def getAverageRepresentation(sentence, T_B, d_lproj):
	sentence = tf.reduce_sum(sentence,reduction_indices=-2)


	sentence_shape = sentence.get_shape().as_list()
	if len(sentence_shape)==2:
		sentence = tf.matmul(sentence,T_B)
	elif len(sentence_shape)==3:
		sentence = tf.reshape(sentence,(-1,sentence_shape[-1]))
		sentence = tf.matmul(sentence,T_B)
		sentence = tf.reshape(sentence,(-1,sentence_shape[1],d_lproj))
	else:
		raise ValueError('Invalid sentence_shape:'+sentence_shape)

	sentence = tf.nn.l2_normalize(sentence,-1)
	return sentence


def getMultiModel(visual_feature, question_feature, answer_feature, common_space_dim):
	'''
		fucntion: getMultiModel
		parameters:
			visual_feature: batch_size * visual_encoded_dim
			question_feature: batch_size * question_encoded_dim
			answer_feature: batch_zize * numberOfChoices * answer_encoded_dim
			common_space_dim: embedding the visual,question,answer to the common space
		return: the embeded vectors(v,q,a)
	'''
	visual_shape = visual_feature.get_shape().as_list()
	question_shape = question_feature.get_shape().as_list()
	answer_shape = answer_feature.get_shape().as_list()

	# build the transformed matrix
	W_v = init_weight_variable((visual_shape[1],common_space_dim),init_method='glorot_uniform',name="W_v")
	W_q = init_weight_variable((question_shape[1],common_space_dim),init_method='glorot_uniform',name="W_q")
	W_a = init_weight_variable((answer_shape[2],common_space_dim),init_method='glorot_uniform',name="W_a")



	answer_feature = tf.reshape(answer_feature,(-1,answer_shape[2]))

	# encoder the features into common space
	T_v = tf.matmul(visual_feature,W_v)
	T_q = tf.matmul(question_feature,W_q)
	T_a = tf.matmul(answer_feature,W_a)

	T_a = tf.reshape(T_a,(-1,answer_shape[1],common_space_dim))

	return T_v,T_q,T_a

def getRankingLoss(T_v, T_q, T_a, answer_index=None, alpha = 0.2 ,isTest=False):
	
	'''
		function: getRankingLoss
		parameters:
			answer_index: the ground truth index, one hot vector
		return:
			loss: tf.float32
	'''
	
	T_v_shape = T_v.get_shape().as_list()
	T_q_shape = T_q.get_shape().as_list()
	T_a_shape = T_a.get_shape().as_list()

	numOfChoices = T_a_shape[1]
	common_space_dim = T_a_shape[2]

	assert T_q_shape == T_v_shape

	T_v = tf.nn.l2_normalize(T_v,1)
	T_q = tf.nn.l2_normalize(T_q,1)
	T_a = tf.nn.l2_normalize(T_a,2)

	T_p = tf.nn.l2_normalize(T_v+T_q,1)

	

	# answer_index = tf.tile(tf.expand_dims(answer_index,dim=-1),[1,1,T_q_shape[-1]]) # sample * numOfChoices * common_space_dim
	

	T_p = tf.tile(tf.expand_dims(T_p,dim=1),[1,numOfChoices,1])

	# T_p = tf.nn.l2_normalize(T_p*T_a,2)
	T_p = T_p*T_a
	T_p = tf.reduce_sum(T_p, reduction_indices=-1)

	scores = T_p

	if not isTest:
		assert answer_index is not None
		positive = tf.reduce_sum(T_p*answer_index, reduction_indices=1, keep_dims=True) # sample , get the positive score
		positive = tf.tile(positive,[1,numOfChoices])

		loss = (alpha - positive + T_p)*(1-answer_index)

		loss = tf.maximum(0.,loss)

		loss = tf.reduce_sum(loss,reduction_indices=-1)

		return loss,scores
	else:
		return scores


def getClassifierLoss(T_s, T_q, T_a, answer_index=None, isTest=False):
	
	'''
		function: getRankingLoss
		parameters:
			answer_index: the ground truth index, one hot vector
		return:
			loss: tf.float32
	'''
	
	T_s_shape = T_s.get_shape().as_list()
	T_q_shape = T_q.get_shape().as_list()
	T_a_shape = T_a.get_shape().as_list()

	numOfChoices = T_a_shape[1]
	common_space_dim = T_a_shape[2]

	assert T_q_shape == T_s_shape

	T_s = tf.nn.l2_normalize(T_s+T_q,1)
	T_a = tf.nn.l2_normalize(T_a,2)

	T_s = tf.tile(tf.expand_dims(T_s,dim=1),[1,numOfChoices,1])

	# T_s = tf.nn.l2_normalize(T_s*T_a,2)
	T_h = T_s*T_a
	T_h = tf.reduce_sum(T_h, reduction_indices=-1)

	scores = T_h

	if not isTest:
		assert answer_index is not None
		loss = tf.nn.softmax_cross_entropy_with_logits(labels = answer_index, logits = scores)
		# acc_value = tf.metrics.accuracy(scores, answer_index)
		return loss,scores
	else:
		return scores



def getVideoSemanticEmbedding(x,w2v,T_B,pca_mat=None):
	'''
		x: input video cnn feature with size of (batch_size, timesteps, channels, height, width)
		w2v: word 2 vec (|v|,dim)
	'''
	input_shape = x.get_shape().as_list()
	w2v_shape = w2v.get_shape().as_list()
	assert(len(input_shape)==5)
	axis = [0,1,3,4,2]
	x = tf.transpose(x,perm=axis)
	x = tf.reshape(x,(-1,input_shape[2]))
	# x = tf.nn.l2_normalize(x,-1)

	if pca_mat is not None:
		linear_proj = tf.Variable(0.1*pca_mat,dtype='float32',name='visual_linear_proj')
	else:
		linear_proj = init_weight_variable((input_shape[2],w2v_shape[-1]), init_method='uniform', name='visual_linear_proj')

	x = tf.matmul(x,linear_proj) 
	x = tf.nn.l2_normalize(x,-1)

	w2v_cov = tf.matmul(tf.transpose(w2v,perm=[1,0]),w2v)

	x = tf.matmul(x,w2v_cov) # (batch_size*timesteps*height*width, |V|)

	x = tf.reshape(x,(-1,input_shape[1],input_shape[3],input_shape[4],w2v_shape[-1]))
	axis = [0,1,4,2,3]
	x = tf.transpose(x,perm=axis)
	
	# can be extended to different architecture
	x = tf.reduce_sum(x,reduction_indices=[1,3,4])
	x = tf.nn.l2_normalize(x,-1)

	x = tf.matmul(x,T_B)

	

	return x



if __name__=='__main__':
	print('video question answering model module!')

	



