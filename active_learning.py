from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config
import itertools
import tensorflow as tf
from random import shuffle
#import gc
import pickle

al = "lu" #most uncertainty (mu) vs. random (rand)
config = Config()
#model = NERModel(config)
#model.build()


def train_active(train, dev, test, select, config):
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
    model.evaluate(test)
    
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
    

def main(i):
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
    out = train_active(train_round, dev, test, select, config)
    #sort select list based on scores

    if al=='mu':
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
    main(int(sys.argv[1]))


