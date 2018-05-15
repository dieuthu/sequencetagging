from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config
import itertools
import tensorflow as tf
from random import shuffle
#import gc
import pickle
from sklearn.cluster import KMeans
import numpy as np


#al = "lu" #most uncertainty (mu) vs. random (rand)
config = Config()
#model = NERModel(config)
#model.build()

def active_strategy(score, transition_params):
    """
    Args: output of CRF
    score: A [seq_len, num_tags] matrix of unary potentials.
    transition_params: A [num_tags, num_tags] matrix of binary potentials.
    """
    trellis = np.zeros_like(score)
    backpointers = np.zeros_like(score, dtype=np.int32)
    trellis[0] = score[0]

    for t in range(1, score.shape[0]):
        v = np.expand_dims(trellis[t - 1], 1) + transition_params
        trellis[t] = score[t] + np.max(v, 0)
        backpointers[t] = np.argmax(v, 0)

    viterbi = [np.argmax(trellis[-1])]
    for bp in reversed(backpointers[1:]):
        viterbi.append(bp[viterbi[-1]])
    viterbi.reverse()

    score = np.max(trellis[-1]) #Score of sequences (higher = better)
    if (self.config.active_strategy=='mg'):
        top_scores = trellis[-1][np.argsort(trellis[-1])[-2:]]
        margin = abs(top_scores[0]-top_scores[1])
        score = margin
    return score

def train_active(train, dev, test, select, config, modename):
    """
    Input: train set, test set, selection set, configurations
    Output: accuracy on dev set, test set, prediction on selection set
    Select Most & Least Certain Examples from Select set
    """
    # build model
    #tf.reset_default_graph()
    #gc.collect()    
    #tf.get_variable_scope().reuse_variables()
    model = NERModel(config)
    model.build()
    print("Start training model...")
    print("Training size ", len(train))
    model.train(train, dev)

    # restore session
    model.restore_session(config.dir_model)

    # evaluate
    print("===Evaluating on test set:===")
    mode = "test" + modename
    model.evaluate(test, mode)
    
    # run on selection set

    print("Selecting samples for active learning...")
    if len(select)==0:
        return []
    l = []
    for sent in select:
        output = model.predict(sent[0])
        l.append(output[1][0])
    #sort l
    return l#most uncertain and least uncertain
    

def main(i, al, filenameextra):
    #Call in an iterator
    # create instance of config
    #config = Config()
    print("********Active training round ", i)
    # Initialize creating dataset
    # create datasets
    train_round = None
    select = None
    dev   = CoNLLDataset(config.filename_dev, config.processing_word,
                                 config.processing_tag, config.max_iter) #always keep the same dev and test

    test  = CoNLLDataset(config.filename_test, config.processing_word,
                                 config.processing_tag, config.max_iter)

    if (i==1):
        train = CoNLLDataset(config.filename_train, config.processing_word,
                         config.processing_tag, config.max_iter)

        train = list(train)
        train_round = train[0:config.num_query]
        select = train[config.num_query:len(train)]
    else:
        fn = open(config.filename_pkl + str(i),'rb')
        train_round, select = pickle.load(fn)
        fn.close()

    print("Training size ", len(train_round))
    print("Number of left training samples ",len(select))
    modename = str(i) + "_" + al + "_" + filenameextra
    out = train_active(train_round, dev, test, select, config, modename)
    #sort select list based on scores

    if config.active_strategy == "cluster":
        print('Scores from cluster ',out)
    else:
        if al=='mu' or al=="mg":
            select = [x for _,x in sorted(zip(out,select))] #Sort based on output of selection
        elif al=='lu':
            select = [x for _,x in sorted(zip(out,select),reverse=True)]
        elif al=='rand':
            shuffle(select)
    
    num_samples = min(config.num_query,len(select))
    train_round+=select[0:num_samples]
    select = select[num_samples:len(select)]
    shuffle(train_round)
    shuffle(select)
    i=i+1
    fo = open(config.filename_pkl+str(i),'wb')
    pickle.dump((train_round,select),fo)
    fo.close()

import sys
if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2], sys.argv[3]) #iter number, active learning algorithm (al), final output name
    #al options: mu (most uncertain), rand (random), lu (least uncertain)

