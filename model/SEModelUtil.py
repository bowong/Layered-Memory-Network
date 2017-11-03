import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
import numpy as np
from sklearn.decomposition import PCA
import ModelUtil
import InitUtil


def ndim(x):
    """Returns the number of axes in a tensor, as an integer.

    # Arguments
        x: Tensor or variable.

    # Returns
        Integer (scalar), number of axes.

    # Examples
    ```python
        >>> from keras import backend as K
        >>> input = K.placeholder(shape=(2, 4, 5))
        >>> val = np.array([[1, 2], [3, 4]])
        >>> kvar = K.variable(value=val)
        >>> K.ndim(input)
        3
        >>> K.ndim(kvar)
        2
    ```
    """
    if isinstance(x, tf.SparseTensor):
        return x._dims

    dims = x.get_shape()._dims
    if dims is not None:
        return len(dims)
    return None
def batch_dot(x, y, axes=None):
    """Batchwise dot product.

    `batch_dot` is used to compute dot product of `x` and `y` when
    `x` and `y` are data in batch, i.e. in a shape of
    `(batch_size, :)`.
    `batch_dot` results in a tensor or variable with less dimensions
    than the input. If the number of dimensions is reduced to 1,
    we use `expand_dims` to make sure that ndim is at least 2.

    # Arguments
        x, y: Keras tensors or variables with `ndim >= 2`
        axes: list of (or single) int with target dimensions.
            The lengths of `axes[0]` and `axes[1]` should be the same.

    # Returns
        A tensor with shape equal to the concatenation of `x`'s shape
        (less the dimension that was summed over) and `y`'s shape
        (less the batch dimension and the dimension that was summed over).
        If the final rank is 1, we reshape it to `(batch_size, 1)`.

    # Examples
        Assume `x = [[1, 2], [3, 4]]` and `y = [[5, 6], [7, 8]]`
        `batch_dot(x, y, axes=1) = [[17, 53]]` which is the main diagonal
        of `x.dot(y.T)`, although we never have to calculate the off-diagonal
        elements.

        Shape inference:
        Let `x`'s shape be `(100, 20)` and `y`'s shape be `(100, 30, 20)`.
        If `axes` is (1, 2), to find the output shape of resultant tensor,
            loop through each dimension in `x`'s shape and `y`'s shape:

        * `x.shape[0]` : 100 : append to output shape
        * `x.shape[1]` : 20 : do not append to output shape,
            dimension 1 of `x` has been summed over. (`dot_axes[0]` = 1)
        * `y.shape[0]` : 100 : do not append to output shape,
            always ignore first dimension of `y`
        * `y.shape[1]` : 30 : append to output shape
        * `y.shape[2]` : 20 : do not append to output shape,
            dimension 2 of `y` has been summed over. (`dot_axes[1]` = 2)
        `output_shape` = `(100, 30)`

    ```python
        >>> x_batch = K.ones(shape=(32, 20, 1))
        >>> y_batch = K.ones(shape=(32, 30, 20))
        >>> xy_batch_dot = K.batch_dot(x_batch, y_batch, axes=[1, 2])
        >>> K.int_shape(xy_batch_dot)
        (32, 1, 30)
    ```
    """
    if isinstance(axes, int):
        axes = (axes, axes)
        #print('1')
    if ndim(x) == 2 and ndim(y) == 2:
        if tf_major_version >= 1:
            if axes[0] == axes[1]:
                out = tf.reduce_sum(tf.multiply(x, y), axes[0])
            else:
                out = tf.reduce_sum(tf.multiply(tf.transpose(x, [1, 0]), y), axes[1])
        else:
            if axes[0] == axes[1]:
                out = tf.reduce_sum(tf.mul(x, y), axes[0])
            else:
                out = tf.reduce_sum(tf.mul(tf.transpose(x, [1, 0]), y), axes[1])
    else:
        if axes is not None:
            #print('2')
            adj_x = None if axes[0] == ndim(x) - 1 else True
            adj_y = True if axes[1] == ndim(y) - 1 else None
        else:
            #print('3')
            adj_x = None
            adj_y = None
        # TODO: remove later.
        if hasattr(tf, 'batch_matmul'):
            try:
                out = tf.batch_matmul(x, y, adj_a=adj_x, adj_b=adj_y)
                #print('4')
            except TypeError:
                out = tf.batch_matmul(x, y, adj_x=adj_x, adj_y=adj_y)
        else:
            out = tf.matmul(x, y, adjoint_a=adj_x, adjoint_b=adj_y)
    if ndim(out) == 1:
        out = expand_dims(out, 1)
    return out


def getVideoDualSemanticEmbeddingWithQuestionAttention(x,w2v,embedded_stories_words,embedded_question,T_B,pca_mat=None):
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

    if pca_mat is not None:
        linear_proj = tf.Variable(0.1*pca_mat,dtype='float32',name='visual_linear_proj')
    else:
        linear_proj = InitUtil.init_weight_variable((input_shape[2],w2v_shape[-1]), init_method='uniform', name='visual_linear_proj')

    x = tf.matmul(x,linear_proj) 
    x = tf.nn.l2_normalize(x,-1)

    w2v_cov = tf.matmul(tf.transpose(w2v,perm=[1,0]),w2v)
    x = tf.matmul(x,w2v_cov) # (batch_size*timesteps*height*width, |V|)

    
    #-----------------------

    x = tf.reshape(x,(-1,input_shape[1],input_shape[3],input_shape[4],w2v_shape[-1]))
    axis = [0,1,4,2,3]
    x = tf.transpose(x,perm=axis)
    
    # can be extended to different architecture
    x = tf.reduce_sum(x,reduction_indices=[3,4])
    x = tf.nn.l2_normalize(x,-1)

    #-----------------------
    stories_cov = batch_dot(tf.transpose(embedded_stories_words,perm=[0,2,1]),embedded_stories_words)
    x = batch_dot(x,stories_cov)
    #-----------------------
    x = tf.nn.l2_normalize(x,-1)

    embedded_question = tf.tile(tf.expand_dims(embedded_question,dim=1),[1,input_shape[1],1])


    frame_weight = tf.reduce_sum(x*embedded_question,reduction_indices=-1,keep_dims=True)
    frame_weight = tf.nn.softmax(frame_weight,dim=1)

    frame_weight =tf.tile(frame_weight,[1,1,w2v_shape[-1]])

    x = tf.reduce_sum(x*frame_weight,reduction_indices=1)

    x = tf.matmul(x,T_B)

    x = tf.nn.l2_normalize(x,-1)
    return x
    
    

def getVideoDualSemanticEmbeddingWithQuestionAttention_up(x,w2v,embedded_stories_words,embedded_question,T_B,pca_mat=None):
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


    if pca_mat is not None:
        linear_proj = tf.Variable(0.1*pca_mat,dtype='float32',name='visual_linear_proj')
    else:
        linear_proj = InitUtil.init_weight_variable((input_shape[2],w2v_shape[-1]), init_method='uniform', name='visual_linear_proj')

    x = tf.matmul(x,linear_proj) 
    x = tf.nn.l2_normalize(x,-1)

    #-----------------------
    w2v_cov = tf.matmul(tf.transpose(w2v,perm=[1,0]),w2v)
    x = tf.matmul(x,w2v_cov) # (batch_size*timesteps*height*width, |V|)

    #-----------------------

    x = tf.reshape(x,(-1,input_shape[1],input_shape[3],input_shape[4],w2v_shape[-1]))
    axis = [0,1,4,2,3]
    x = tf.transpose(x,perm=axis)
    
    # can be extended to different architecture
    x = tf.reduce_sum(x,reduction_indices=[3,4])
    x = tf.nn.l2_normalize(x,-1)
    
    

    #-----------------------


    stories_cov = batch_dot(tf.transpose(embedded_stories_words,perm=[0,2,1]),embedded_stories_words)
    x_out = batch_dot(x,stories_cov)


    
    #-----------------------
    x = tf.nn.l2_normalize(x_out,-1)

    embedded_question_use = tf.tile(tf.expand_dims(embedded_question,dim=1),[1,input_shape[1],1])

    
    frame_weight = tf.reduce_sum(x*embedded_question_use,reduction_indices=-1,keep_dims=True)
    
    frame_weight = tf.nn.softmax(frame_weight,dim=1) 
    


    frame_weight =tf.tile(frame_weight,[1,1,w2v_shape[-1]])

    x_weight_new = tf.reduce_sum(x*frame_weight,reduction_indices=1)
    

    x_weight_use = tf.expand_dims(x_weight_new, dim = 1)
    
    story_weight = tf.matmul(x_weight_use,tf.transpose(embedded_stories_words,perm=[0,2,1]))
    
    story_weight = tf.nn.relu(story_weight)
    
    embedded_stories_words = tf.multiply(tf.transpose(story_weight,perm=[0,2,1]), embedded_stories_words)
    stories_cov = batch_dot(tf.transpose(embedded_stories_words,perm=[0,2,1]),embedded_stories_words)
    
    x = batch_dot(x,stories_cov)


    

    x = tf.nn.l2_normalize(x,-1)
    

    frame_weight = tf.reduce_sum(x*embedded_question_use,reduction_indices=-1,keep_dims=True)
    
    frame_weight = tf.nn.softmax(frame_weight,dim=1) 

    frame_weight =tf.tile(frame_weight,[1,1,w2v_shape[-1]])

    x = tf.reduce_sum(x*frame_weight,reduction_indices=1)
    

    x = tf.matmul(x,T_B)

    x = tf.nn.l2_normalize(x,-1)
    return x


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





def getMemoryNetworks(embeded_stories, embeded_question, d_lproj, return_sequences=False):

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


    embeded_stories = tf.reduce_sum(embeded_stories,reduction_indices=-2)
    embeded_stories = tf.nn.l2_normalize(embeded_stories,-2)

    
    embeded_question = tf.tile(tf.expand_dims(embeded_question,dim=1),[1,num_of_sentence,1])

    sen_weight = tf.reduce_sum(embeded_question*embeded_stories,reduction_indices=-1,keep_dims=True)

    sen_weight = tf.nn.softmax(sen_weight,dim=1)
    sen_weight = tf.tile(sen_weight,[1,1,output_dims])
    if return_sequences:
        embeded_stories = embeded_stories*sen_weight
    else:
        embeded_stories = tf.reduce_sum(embeded_stories*sen_weight,reduction_indices=1) # (batch_size, output_dims)

    return embeded_stories



    
def getVideoDualSemanticEmbeddingWithQuestionAttention_question_guid(embeded_stories, d_lproj, x,w2v,embedded_stories_words,embedded_question,T_B,pca_mat=None,return_sequences=True):
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

    if pca_mat is not None:
        linear_proj = tf.Variable(0.1*pca_mat,dtype='float32',name='visual_linear_proj')
    else:
        linear_proj = InitUtil.init_weight_variable((input_shape[2],w2v_shape[-1]), init_method='uniform', name='visual_linear_proj')

    x = tf.matmul(x,linear_proj) 
    x = tf.nn.l2_normalize(x,-1)


    
    #-----------------------
    w2v_cov = tf.matmul(tf.transpose(w2v,perm=[1,0]),w2v)
    x = tf.matmul(x,w2v_cov) # (batch_size*timesteps*height*width, |V|)

    #-----------------------

    x = tf.reshape(x,(-1,input_shape[1],input_shape[3],input_shape[4],w2v_shape[-1]))
    axis = [0,1,4,2,3]
    x = tf.transpose(x,perm=axis)
    
    # can be extended to different architecture
    x = tf.reduce_sum(x,reduction_indices=[3,4])
    x = tf.nn.l2_normalize(x,-1)
    
    

    #-----------------------

    stories_cov = batch_dot(tf.transpose(embedded_stories_words,perm=[0,2,1]),embedded_stories_words)
    x_out = batch_dot(x,stories_cov)


    
    #-----------------------
    x = tf.nn.l2_normalize(x_out,-1)

    embedded_question_use = tf.tile(tf.expand_dims(embedded_question,dim=1),[1,input_shape[1],1])

    
    frame_weight = tf.reduce_sum(x*embedded_question_use,reduction_indices=-1,keep_dims=True)
    
    frame_weight = tf.nn.softmax(frame_weight,dim=1) 
    


    frame_weight =tf.tile(frame_weight,[1,1,w2v_shape[-1]])

    x_weight_new = tf.reduce_sum(x*frame_weight,reduction_indices=1)
    

    x_weight_use = tf.expand_dims(x_weight_new, dim = 1)
    
    story_weight = tf.matmul(x_weight_use,tf.transpose(embedded_stories_words,perm=[0,2,1]))
    
    story_weight = tf.nn.relu(story_weight)
    
    embedded_stories_words = tf.multiply(tf.transpose(story_weight,perm=[0,2,1]), embedded_stories_words)
    
    stories_cov = batch_dot(tf.transpose(embedded_stories_words,perm=[0,2,1]),embedded_stories_words)
    
    x = batch_dot(x,stories_cov)
    
    x = tf.nn.l2_normalize(x_out,-1)
    
    #-------------------------------------------------------------------------------------------------------------
    
    stories_shape = embeded_stories.get_shape().as_list()
    embeded_question_shape = embedded_question.get_shape().as_list()
    num_of_sentence = stories_shape[-3]
    input_dims = stories_shape[-1]
    output_dims = embeded_question_shape[-1]
    
    print('embeded_question_shape', embeded_question_shape)
    print('num_of_sentence', num_of_sentence)
    
    print('output_dims', output_dims)
    print('stories_shape', stories_shape)

    
    embeded_question = tf.tile(tf.expand_dims(embedded_question,dim=1),[1,num_of_sentence,1])

    sen_weight = tf.reduce_sum(embeded_question*embedded_stories_words,reduction_indices=-1,keep_dims=True)


    sen_weight = tf.nn.relu(sen_weight)
    sen_weight = tf.tile(sen_weight,[1,1,output_dims])
    if return_sequences:
        embeded_stories_used = embedded_stories_words*sen_weight
    else:
        embeded_stories_used = tf.reduce_sum(embedded_stories_words*sen_weight,reduction_indices=1)
 
    
    #-------------------------------------------------------------------------------------------------------------
    stories_cov = batch_dot(tf.transpose(embeded_stories_used,perm=[0,2,1]),embeded_stories_used)
    
    x = batch_dot(x,stories_cov)


    

    #-----------------------
    x = tf.nn.l2_normalize(x,-1)
    


    
    frame_weight = tf.reduce_sum(x*embedded_question_use,reduction_indices=-1,keep_dims=True)
    
    frame_weight = tf.nn.softmax(frame_weight,dim=1) 

    frame_weight =tf.tile(frame_weight,[1,1,w2v_shape[-1]])

    x = tf.reduce_sum(x*frame_weight,reduction_indices=1)


    #-----------------------------------------------

    x = tf.matmul(x,T_B)

    x = tf.nn.l2_normalize(x,-1)
    
    return x


