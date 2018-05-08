from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config
import itertools


def train_active(train, test, select, config):
    """
    Input: train set, test set, selection set, configurations
    Output: accuracy on dev set, test set, prediction on selection set
    Select Most & Least Certain Examples from Select set
    """
    # build model
    model = NERModel(config)
    model.build()
    model.train(train_init, dev)

    # restore session
    model.restore_session(config.dir_model)

    # evaluate
    print("===Evaluating on test set:===")
    model.evaluate(test)
    
    # run on selection set

    print("Selecting samples for active learning...")    
    output = model.evaluate()
    
    return (output["mu"], output["lu"])
    

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
    train_round = train[0:config.num_query]
    select = train[config.num_query:len(train)]
    
    i = 1
    while True:
        if (len(select)>config.num_query):            
            print("Active training round ", i)
            out = train_active(train_round, test, select, config)
            train_round = itertools.chain([train_round],out["mu"]) #add MU to train_round            
            #train_round += out["mu"]
            select = list(select)
            for i in out["mu"]:
                select = select.remove(i)#Remove MU from select set
            select = iter(select)
            i+=1
        else:
            break
    if len(select)>0:
        print("Last Active training round ", i)
        out = train_active(train_round, test, select, config)        
    
    
if __name__ == "__main__":
    main()
