import numpy as np
import os
#import MovieQA_benchmark as MovieQA
import re
import h5py
import tensorflow as tf
import math
from nltk.stem.snowball import SnowballStemmer
from collections import Counter

re_alphanumeric = re.compile('[^a-z0-9 -]+')
re_multispace = re.compile(' +')
snowball = SnowballStemmer('english') 

def preprocess_sentence(line):
    '''strip all punctuation, keep only alphanumerics
    '''
    line = re_alphanumeric.sub('', line)
    line = re_multispace.sub(' ', line)
    return line

def normalize_documents(stories, v2i, max_words=40):
    """Normalize all stories in the dictionary, get list of words per sentence.
    """
    for movie in stories.keys():
        for s, sentence in enumerate(stories[movie]):
            sentence = sentence.lower()
            sentence = preprocess_sentence(sentence.strip())
            sentence = sentence.split(' ')[:max_words]
            stories[movie][s] = sentence

    max_sentences = max([len(story) for story in stories.values()])
    max_words = max([len(sent) for story in stories.values() for sent in story])

    processed_stories = {}
    for imdb_key, story in stories.items():
        processed_stories[imdb_key] = np.zeros((max_sentences,max_words), dtype='int32')
        for jj, sentence in enumerate(story):
            for kk, word in enumerate(sentence):
                if v2i.has_key(word):
                    processed_stories[imdb_key][jj, kk] = v2i[word]
                else:
                    processed_stories[imdb_key][jj, kk] = v2i['UNK']

    return processed_stories,max_sentences,max_words

def preprocess_stories(stories,max_words=40):
    for movie in stories.keys():
        for s, sentence in enumerate(stories[movie]):
            sentence = sentence.lower()
            sentence = preprocess_sentence(sentence)
            sentence = sentence.split(' ')[:max_words]
            stories[movie][s] = sentence
    return stories

def create_vocabulary(QAs, stories, word_thresh=2, v2i={'': 0, 'UNK':1}):
    '''
    v2i = {'': 0, 'UNK':1}  # vocabulary to index
    '''
    print 'Create vocabulary...'

    # Get all story words
    all_words = [word for story in stories for sent in story for word in sent]
    print('number of words: %d' %len(all_words))


    QA_words = {}
    for QA in QAs:
        temp = {}
        q_w = preprocess_sentence(QA.question.strip().lower()).split(' ')
        a_w = [preprocess_sentence(answer.strip().lower()).split(' ') for answer in QA.answers]
        temp['q_w'] = q_w
        temp['a_w'] = a_w
        temp['qid'] = QA.qid
        temp['imdb_key'] = QA.imdb_key
        temp['question'] = QA.question
        temp['answers'] = QA.answers
        temp['correct_index'] = QA.correct_index
        # temp['plot_alignment'] = QA.plot_alignment
        temp['video_clips'] = QA.video_clips

        
        QA_words[QA.qid]=temp

        all_words.extend(q_w)
        for answer in a_w:
            all_words.extend(answer)


    # threshold vocabulary, at least N instances of every word
    vocab = Counter(all_words)
    vocab = [k for k in vocab.keys() if vocab[k] >= word_thresh]

    # create vocabulary index
    for w in vocab:
        if w not in v2i.keys():
            v2i[w] = len(v2i)
    
    print('Created a vocabulary of %d words. Threshold removed %.2f %% words'\
        %(len(v2i), 100*(1. * len(set(all_words))-len(v2i))/len(all_words)))

    return QA_words, v2i

def create_vocabulary_word2vec(QAs, stories, word_thresh=2, w2v_vocab=None, v2i={'': 0, 'UNK':1}):
    '''
    v2i = {'': 0, 'UNK':1}  # vocabulary to index
    '''
    print 'Create vocabulary...'

    if w2v_vocab is not None:
        print "Adding words based on word2vec"
    else:    
        print "Adding all words"

    # Get all story words
    all_words = [word for story in stories for sent in story for word in sent]
    print('number of total words: %d' %len(all_words))


    for QA in QAs:
        q_w = preprocess_sentence(QA.question.strip().lower()).split(' ')
        a_w = [preprocess_sentence(answer.strip().lower()).split(' ') for answer in QA.answers]
        
        all_words.extend(q_w)
        for answer in a_w:
            all_words.extend(answer)


    # threshold vocabulary, at least N instances of every word
    vocab = Counter(all_words)
    vocab = [k for k in vocab.keys() if vocab[k] >= word_thresh]

    # create vocabulary index
    for w in vocab:
        if w not in v2i.keys():
            if w2v_vocab is None:
                # if word2vec is not provided, just dump the word to vocab
                v2i[w] = len(v2i)
            elif w2v_vocab is not None and w in w2v_vocab:
                # check if word in vocab, or else ignore
                v2i[w] = len(v2i)
    
    print('Created a vocabulary of %d words. Threshold removed %.2f %% words'\
        %(len(v2i), 100*(1. * len(set(all_words))-len(v2i))/len(all_words)))

    return v2i

def data_in_matrix_form(stories, v2i,max_sentences=None,max_words=None):
    """Make the QA data set compatible for memory networks by
    converting to matrix format (index into LUT vocabulary).
    """

    def add_word_or_UNK():
        if v2i.has_key(word):
            return v2i[word]
        else:
            return v2i['UNK']

    # Encode stories
    if max_sentences is None:
        max_sentences = max([len(story) for story in stories.values()])
    if max_words is None:
        max_words = max([len(sent) for story in stories.values() for sent in story])

    storyM = {}
    for imdb_key, story in stories.iteritems():
        storyM[imdb_key] = np.zeros((max_sentences, max_words), dtype='int32')
        for jj, sentence in enumerate(story):
            for kk, word in enumerate(sentence):
                storyM[imdb_key][jj, kk] = add_word_or_UNK()

    print "#stories:", len(storyM)
    print "storyM shape (movie 1):", storyM.values()[0].shape

    
    return storyM,max_sentences,max_words



def S2I(sen, v2i, fixed_len):
    '''
        len_qa: fixed length of question or answer
    '''
    if type(sen)!=list:
        sen = preprocess_sentence(sen.strip().lower()).split(' ')
    res = []
    for idx, w in enumerate(sen):
        if idx<fixed_len:
            if w in v2i.keys():
                res.append(v2i[w])
            else:
                res.append(v2i['UNK'])
    while(len(res)<fixed_len):
        res.append(v2i[''])
    return res


def getBatchIndexedQAs(batch_qas_list,v2i, nql=16, nqa=10, numOfChoices=2):
    '''
        batch_qas_list: list of qas
        QA_words: all the QAs, contains question words and answer words
        v2i: vocabulary to index
        nql: length of question
        nqa: length of answer
        numOfChoices: number of Choices utilized per QA, default set to 2 ==> right/wrong

        return: questions, answers, ground_truth
            both of them are numeric indexed
            ground_truth is one hot vector
    '''

    batch_size = len(batch_qas_list)
    questions = np.zeros((batch_size,nql),dtype='int32')
    answers = np.zeros((batch_size,numOfChoices,nqa),dtype='int32')
    ground_truth = np.zeros((batch_size,numOfChoices),dtype='int32')

    for idx, qa in enumerate(batch_qas_list):
        # set question 
        qid = qa.qid
        questions[idx][:]=S2I(qa.question, v2i,nql)
        
        
        # set anwsers
        if numOfChoices==2:
            ground_answer_pos = np.random.randint(0,numOfChoices)
            ground_truth[idx][ground_answer_pos]=1
            
            # set correct answer
            correct_index = int(qa.correct_index)
            answers[idx][ground_answer_pos][:] = S2I(qa.answers[correct_index], v2i, nqa)



            wrong_index = np.random.randint(0,5)
            while(wrong_index==correct_index):
                wrong_index = np.random.randint(0,5)

            # set wrong answer
            answers[idx][1-ground_answer_pos][:]=S2I(qa.answers[wrong_index], v2i, nqa)
        elif numOfChoices==5:
            
            # set correct answer
            correct_index = int(qa.correct_index)
            ground_truth[idx][correct_index]=1
            for ans_idx, ans in enumerate(qa.answers):
                answers[idx][ans_idx][:]=S2I(ans, v2i, nqa)

        else:
            raise ValueError('Invalid numOfChoices: ' + numOfChoices)

    return questions,answers,ground_truth

def getBatchTestIndexedQAs(batch_qas_list,v2i, nql=16, nqa=10, numOfChoices=2):
    '''
        batch_qas_list: list of qas
        v2i: vocabulary to index
        nql: length of question
        nqa: length of answer
        numOfChoices: number of Choices utilized per QA, default set to 2 ==> right/wrong

        return: questions, answers, ground_truth
            both of them are numeric indexed
            ground_truth is one hot vector
    '''

    batch_size = len(batch_qas_list)
    questions = np.zeros((batch_size,nql),dtype='int32')
    answers = np.zeros((batch_size,numOfChoices,nqa),dtype='int32')

    for idx, qa in enumerate(batch_qas_list):
        # set question 
        qid = qa.qid
        questions[idx][:]=S2I(qa.question, v2i,nql)
        
        # set anwsers
        for ans_idx, ans in enumerate(qa.answers):
            answers[idx][ans_idx][:]=S2I(ans, v2i, nqa)


    return questions,answers

def getBatchVideoFeature(batch_qas_list, hf, feature_shape):
    '''
        video-based QA
        there are video clips in all QA pairs.  
    '''

    batch_size = len(batch_qas_list)
    input_video = np.zeros((batch_size,)+tuple(feature_shape),dtype='float32')

    timesteps = feature_shape[0]

    for idx, qa in enumerate(batch_qas_list):
        qid = qa.qid
        video_clips = qa.video_clips
        imdb_key = qa.imdb_key



        clips_features = []
        if len(video_clips) != 0:
            for clip in video_clips:
                dataset = imdb_key+'/'+clip
                if imdb_key in hf.keys() and clip in hf[imdb_key].keys():
                    clips_features.extend(hf[dataset][:]) # clips_features.shape


        if(len(clips_features)<=0):
            # if there are not vlid features
            for clip in hf[imdb_key].keys():
                dataset = imdb_key+'/'+clip
                clips_features.extend(hf[dataset][:]) # clips_features.shape

        
        if(len(clips_features)>=timesteps):
            interval = int(math.floor((len(clips_features)-1)/(timesteps-1)))
            input_video[idx] = clips_features[0::interval][0:timesteps]
        else:
            input_video[idx][:len(clips_features)] = clips_features
            for last_idx in xrange(len(clips_features),timesteps):
                input_video[idx][last_idx]=clips_features[-1]


        # if qid not in hf_out.keys():
        #     dset = hf_out.create_dataset(qid, feature_shape, dtype='f')
        #     dset[:] = input_video[idx]


    return input_video

def getBatchVideoFeatureFromQid(batch_qas_list, hf, feature_shape):
    '''
        video-based QA
        there are video clips in all QA pairs.  
    '''

    batch_size = len(batch_qas_list)
    input_video = np.zeros((batch_size,)+tuple(feature_shape),dtype='float32')

    timesteps = feature_shape[0]
    for idx, qa in enumerate(batch_qas_list):
        qid = qa.qid
        input_video[idx] = hf[qid][:]
    return input_video
    
rng = np.random
rng.seed(1234)
def getw2v(batch_qa_list,w2v,v2i,d_w2v = 300):


    voc_size = len(v2i)


    pca_mat = None
    #print "Initialize LUTs as word2vec and use linear projection layer"


    LUT = np.zeros((voc_size, d_w2v), dtype='float32')
    found_words = 0

    for w, v in v2i.iteritems():
        if w in w2v.vocab:
            LUT[v] = w2v.get_vector(w)
            found_words +=1
        else:
            LUT[v] = rng.randn(d_w2v)
            LUT[v] = LUT[v] / (np.linalg.norm(LUT[v]) + 1e-6)

    #print "Found %d / %d words" %(found_words, len(v2i))


    # word 0 is blanked out, word 1 is 'UNK'
    LUT[0] = np.zeros((d_w2v))

    # if linear projection layer is not the same shape as LUT, then initialize with PCA


    # setup LUT!
    T_w2v = tf.constant(LUT.astype('float32'))


    word_shape = (26033, 300)

    w2v_new = np.zeros((batch_qa_list,)+word_shape,dtype='int32')


    for idx in xrange(batch_qa_list):
    

    
    
        w2v_new[idx][:] = LUT[:]

    return w2v_new

        #w2v_new = tf.tile(w2v_new, [input_shape[0],1,1])
    

def getBatchIndexedStories(batch_qa_list,stories,v2i,story_shape):
    batch_size = len(batch_qa_list)
    input_stories = np.zeros((batch_size,)+story_shape,dtype='int32')

    for idx, qa in enumerate(batch_qa_list):
        imdb_key = qa.imdb_key
        interval = int(math.floor((len(stories[imdb_key])-1)/(story_shape[0]-1)))

        if interval != 0:
            for k in xrange(story_shape[0]):
                # if(k<story_shape[0]):
                input_stories[idx][k] = stories[imdb_key][k*interval,:story_shape[1]]
        else:
            input_stories[idx][:len(stories[imdb_key])] = stories[imdb_key][:,:story_shape[1]]

    return input_stories

def getBatchRfrVideoFeature(batch_qa, hf, feature_shape,false_frame_num=2):
    batch_size = len(batch_qa)
    input_video = np.zeros((batch_size,)+tuple(feature_shape),dtype='float32')

    timesteps = feature_shape[0]
    rfr_lables = np.zeros((batch_size,timesteps,2),dtype='int32')
    rfr_lables[:,:,1] = 1
    
    for idx, qa in enumerate(batch_qa):
        qid = qa.qid
        video_clips = qa.video_clips
        imdb_key = qa.imdb_key
        clips_features = []
        false_clips_features = []
        if len(video_clips) != 0:
            for clip in video_clips:
                dataset = imdb_key+'/'+clip
                if imdb_key in hf.keys() and clip in hf[imdb_key].keys():
                    clips_features.extend(hf[dataset][:]) # clips_features.shape
                

        for clip in hf[imdb_key].keys():
            dataset = imdb_key+'/'+clip
            if clip not in video_clips:
                false_clips_features.extend(hf[dataset][:])


        if(len(clips_features)<=0):
            # if there are not vlid features
            for clip in hf[imdb_key].keys():
                dataset = imdb_key+'/'+clip
                clips_features.extend(hf[dataset][:]) # clips_features.shape

        
        if(len(clips_features)>=timesteps):
            interval = int(math.floor((len(clips_features)-1)/(timesteps-1)))
            input_video[idx] = clips_features[0::interval][0:timesteps]
        else:
            input_video[idx][:len(clips_features)] = clips_features
            for last_idx in xrange(len(clips_features),timesteps):
                input_video[idx][last_idx]=clips_features[-1]

        false_clips_features = np.random.permutation(false_clips_features)

        false_frame_pos = np.random.permutation(range(0,timesteps))[:false_frame_num]
        for _,ffp in enumerate(false_frame_pos):
            input_video[idx][ffp] = false_clips_features[ffp]
            rfr_lables[idx,ffp,0] = 1
            rfr_lables[idx,ffp,1] = 0

    return input_video, rfr_lables


def split_stories(full_stories,train_movies,val_movies):
    train_stories = {}
    val_stories = {}
    for tm in train_movies:
        train_stories[tm] = full_stories[tm]
    for vm in val_movies:
        val_stories[vm] = full_stories[vm]

    print('num of train stories:',len(train_stories))
    print('num of val stories:',len(val_stories))
    return train_stories,val_stories

def getBatchIndexedQAs_return(batch_qas_list,v2i, nql=16, nqa=10, numOfChoices=2):
    '''
        batch_qas_list: list of qas
        QA_words: all the QAs, contains question words and answer words
        v2i: vocabulary to index
        nql: length of question
        nqa: length of answer
        numOfChoices: number of Choices utilized per QA, default set to 2 ==> right/wrong

        return: questions, answers, ground_truth
            both of them are numeric indexed
            ground_truth is one hot vector
    '''

    batch_size = len(batch_qas_list)
    questions = np.zeros((batch_size,nql),dtype='int32')
    answers = np.zeros((batch_size,numOfChoices,nqa),dtype='int32')
    ground_truth = np.zeros((batch_size,numOfChoices),dtype='int32')

    for idx, qa in enumerate(batch_qas_list):
        # set question 

        questions[idx][:]=qa.question
        
        if numOfChoices==5:
            
            # set correct answer
            #correct_index = qa.correct_index
            ground_truth[idx]=qa.correct_index
            for ans_idx, ans in enumerate(qa.answers):
                answers[idx][ans_idx][:]=ans

    
        else:
            raise ValueError('Invalid numOfChoices: ' + numOfChoices)

    return questions,answers,ground_truth

def getTestBatchIndexedQAs_return(batch_qas_list,v2i, nql=16, nqa=10, numOfChoices=2):

    batch_size = len(batch_qas_list)
    questions = np.zeros((batch_size,nql),dtype='int32')
    answers = np.zeros((batch_size,numOfChoices,nqa),dtype='int32')

    for idx, qa in enumerate(batch_qas_list):

        questions[idx][:]=qa.question
        
        if numOfChoices==5:
            
            for ans_idx, ans in enumerate(qa.answers):
                answers[idx][ans_idx][:]=ans
        else:
            raise ValueError('Invalid numOfChoices: ' + numOfChoices)

    return questions,answers  
def main():
    
    task = 'video-based' # video-based or subtitle-based

    mqa = MovieQA.DataLoader()


    # get 'subtitile-based' QA task dataset
    stories, subtitle_QAs = mqa.get_story_qa_data('train', 'subtitle')

    # Create vocabulary
    QA_words, v2i = create_vocabulary(subtitle_QAs, stories, word_thresh=2, v2i={'': 0, 'UNK':1})

    # get 'video-based' QA task training set
    vl_qa, video_QAs = mqa.get_video_list('train', 'qa_clips')  # key: 'train:<id>', value: list of related clips
    # vl_qa, _ = mqa.get_video_list('train', 'all_clips') # key:moive vid, value:list of related movid all_clips


    
    all_video_train_list = video_QAs

    batch_size = 20
    total_train_qa = len(all_video_train_list)
    num_batch = int(round(total_train_qa*1.0/batch_size))

    total_epoch = 100

    hf = h5py.File('/home/wb/movie_feature.hdf5','r')
    feature_shape = (10,1024)
    for epoch in xrange(total_epoch):
        #shuffle
        np.random.shuffle(all_video_train_list)
        for batch_idx in xrange(num_batch):
            batch_qa = all_video_train_list[batch_idx*batch_size:min((batch_idx+1)*batch_size,total_train_qa)]
            questions,answers,ground_truth = getBatchIndexedQAs(batch_qa,QA_words,v2i, nql=16, nqa=10, numOfChoices=2)
            input_video = getBatchVideoFeature(batch_qa, QA_words, hf, feature_shape)
            print(input_video)
            print(ground_truth)
            break
        break


if __name__=='__main__':
    main()