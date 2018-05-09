from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config
import itertools
import tensorflow as tf

al = "random"

def train_active(train, dev, test, select, config):
    """
    Input: train set, test set, selection set, configurations
    Output: accuracy on dev set, test set, prediction on selection set
    Select Most & Least Certain Examples from Select set
    """
    # build model
    tf.reset_default_graph()
    model = NERModel(config)
    model.build()
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
    

def main():
    # create instance of config
    config = Config()

    # Initialize creating dataset
    # create datasets
    dev   = CoNLLDataset(config.filename_dev, config.processing_word,
                         config.processing_tag, config.max_iter) #always keep the same dev and test
                         
    train = CoNLLDataset(config.filename_train, config.processing_word,
                         config.processing_tag, config.max_iter)

    test  = CoNLLDataset(config.filename_test, config.processing_word,
                         config.processing_tag, config.max_iter)

    # train model with each round
    train = list(train)
    train_round = train[0:config.num_query]
    select = train[config.num_query:len(train)]
    
    i = 1
    while True:
        if (len(select)>config.num_query):            
            print("********Active training round ", i)
            print("Training size ", len(train_round))
            print("Number of left training samples ",len(select))
            out = train_active(train_round, dev, test, select, config)
            #sort select list based on scores

            if al=='mu':
                select = [x for _,x in sorted(zip(out,select))] #Sort based on output of selection
            elif al=='lu':
                select = [x for _,x in sorted(zip(out,select),reverse=True)]
            train_round+=select[0:config.num_query]
            select = select[config.num_query:len(select)]

            #train_round = itertools.chain([train_round],out["mu"]) #add MU to train_round           
            #for sent in out[0]:
            #    train_round.append(sent[0])
            #    select = select.remove(sent[0])#Remove MU from select set
            #select = iter(select)
            i+=1
        else:
            break
    if len(select)>0:
        print("Last Active training round ", i)
        print("Training size ", len(train))
        print("Number of left training samples ",len(select))
        out = train_active(train, dev, test, [], config)        
    
    
if __name__ == "__main__":
    main()

