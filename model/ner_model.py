import pdb
import numpy as np
import os
import tensorflow as tf
import math

from .data_utils import minibatches, pad_sequences, get_chunks
from .general_utils import Progbar
from .base_model import BaseModel


class NERModel(BaseModel):
    """Specialized class of Model for NER"""

    def __init__(self, config):
        super(NERModel, self).__init__(config)
        self.idx_to_tag = {idx: tag for tag, idx in
                           self.config.vocab_tags.items()}
        self.tag_to_idx = {tag: idx for tag, idx in
                            self.config.vocab_tags.items()}
        
    def add_placeholders(self):
        """Define placeholders = entries to computational graph"""
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_ids")

        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                        name="sequence_lengths")

        # shape = (batch size, max length of sentence, max length of word)
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None],
                        name="char_ids")

        # shape = (batch_size, max_length of sentence)
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None],

                        name="word_lengths")

        # shape = (batch size, max length of sentence in batch)
        self.labels = tf.placeholder(tf.int32, shape=[None, None],
                        name="labels")

        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                        name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                        name="lr")


    def get_feed_dict(self, words, labels=None, lr=None, dropout=None):
        """Given some data, pad it and build a feed dictionary

        Args:
            words: list of sentences. A sentence is a list of ids of a list of
                words. A word is a list of ids
            labels: list of ids
            lr: (float) learning rate
            dropout: (float) keep prob

        Returns:
            dict {placeholder: value}

        """
        # perform padding of the given data
        if self.config.use_chars:
            char_ids, word_ids = zip(*words)
            word_ids, sequence_lengths = pad_sequences(word_ids, 0)
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0,
                nlevels=2)
        else:
            word_ids, sequence_lengths = pad_sequences(words, 0)

        # build feed dictionary
        feed = {
            self.word_ids: word_ids,
            self.sequence_lengths: sequence_lengths
        }

        if self.config.use_chars:
            feed[self.char_ids] = char_ids
            feed[self.word_lengths] = word_lengths

        if labels is not None:
            labels, _ = pad_sequences(labels, 0)
            feed[self.labels] = labels

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed, sequence_lengths


    def add_word_embeddings_op(self):
        """Defines self.word_embeddings

        If self.config.embeddings is not None and is a np array initialized
        with pre-trained word vectors, the word embeddings is just a look-up
        and we don't train the vectors. Otherwise, a random matrix with
        the correct shape is initialized.
        """
        with tf.variable_scope("words"):
            if self.config.embeddings is None:
                self.logger.info("WARNING: randomly initializing word vectors")
                _word_embeddings = tf.get_variable(
                        name="_word_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nwords, self.config.dim_word])
            else:
                _word_embeddings = tf.Variable(
                        self.config.embeddings,
                        name="_word_embeddings",
                        dtype=tf.float32,
                        trainable=self.config.train_embeddings)

            word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                    self.word_ids, name="word_embeddings")

        with tf.variable_scope("chars"):
            if self.config.use_chars:
                # get char embeddings matrix
                _char_embeddings = tf.get_variable(
                        name="_char_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nchars, self.config.dim_char])
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                        self.char_ids, name="char_embeddings")

                # put the time dimension on axis=1
                s = tf.shape(char_embeddings)
                char_embeddings = tf.reshape(char_embeddings,
                        shape=[s[0]*s[1], s[-2], self.config.dim_char])
                word_lengths = tf.reshape(self.word_lengths, shape=[s[0]*s[1]])

                # bi lstm on chars
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                        state_is_tuple=True)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                        state_is_tuple=True)
                _output = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, char_embeddings,
                        sequence_length=word_lengths, dtype=tf.float32)

                # read and concat output
                _, ((_, output_fw), (_, output_bw)) = _output
                output = tf.concat([output_fw, output_bw], axis=-1)

                # shape = (batch size, max sentence length, char hidden size)
                output = tf.reshape(output,
                        shape=[s[0], s[1], 2*self.config.hidden_size_char])
                word_embeddings = tf.concat([word_embeddings, output], axis=-1)

        self.word_embeddings =  tf.nn.dropout(word_embeddings, self.dropout)


    def add_logits_op(self):
        """Defines self.logits

        For each word in each sentence of the batch, it corresponds to a vector
        of scores, of dimension equal to the number of tags.
        """
        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.word_embeddings,
                    sequence_length=self.sequence_lengths, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout)

        with tf.variable_scope("proj"):
            W = tf.get_variable("W", dtype=tf.float32,
                    shape=[2*self.config.hidden_size_lstm, self.config.ntags])

            b = tf.get_variable("b", shape=[self.config.ntags],
                    dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2*self.config.hidden_size_lstm])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, nsteps, self.config.ntags])


    def add_pred_op(self):
        """Defines self.labels_pred

        This op is defined only in the case where we don't use a CRF since in
        that case we can make the prediction "in the graph" (thanks to tf
        functions in other words). With theCRF, as the inference is coded
        in python and not in pure tensroflow, we have to make the prediciton
        outside the graph.
        """
        if not self.config.use_crf:
            self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1),
                    tf.int32)


    def add_loss_op(self):
        """Defines the loss"""
        if self.config.use_crf:
            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                    self.logits, self.labels, self.sequence_lengths)
            self.trans_params = trans_params # need to evaluate it for decoding
            self.loss = tf.reduce_mean(-log_likelihood)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.logits, labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

        # for tensorboard
        tf.summary.scalar("loss", self.loss)


    def build(self):
        # NER specific functions
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_logits_op()
        self.add_pred_op()
        self.add_loss_op()

        # Generic functions that add training op and initialize session
        self.add_train_op(self.config.lr_method, self.lr, self.loss,
                self.config.clip)
        self.initialize_session() # now self.sess is defined and vars are init


    def predict_batch(self, words):
        """
        Args:
            words: list of sentences

        Returns:
            labels_pred: list of labels for each sentence
            sequence_length

        """
        fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0)

        if self.config.use_crf:
            # get tag scores and transition params of CRF
            viterbi_sequences = []
            scores = []

            logits, trans_params = self.sess.run(
                    [self.logits, self.trans_params], feed_dict=fd)

            #logits = sigmoid_v(logits)
            #trans_params = sigmoid_v(trans_params)
            # iterate over the sentences because no batching in vitervi_decode
            for logit, sequence_length in zip(logits, sequence_lengths):
                logit = logit[:sequence_length] # keep only the valid steps
                #print("Logit ", logit)
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                        logit, trans_params)
                viterbi_sequences += [viterbi_seq]
                #print('trans_params ', trans_params)
                #print('Sequence ', viterbi_seq)
                #print(sequence_length)
                #print(len(viterbi_seq))
                #print('Score ', viterbi_score)#Use to decide least-uncertainty
                if self.config.active_strategy=="nus": 
                    viterbi_score = float(viterbi_score/sequence_length)
                else:
                    viterbi_score = active_strategy(logit, trans_params, self.config.active_strategy, self.tag_to_idx)
                scores.append(viterbi_score)
            return viterbi_sequences, sequence_lengths, scores

        else:
            labels_pred = self.sess.run(self.labels_pred, feed_dict=fd)

            return labels_pred, sequence_lengths, None


    def run_epoch(self, train, dev, epoch):
        """Performs one complete pass over the train set and evaluate on dev

        Args:
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            epoch: (int) index of the current epoch

        Returns:
            f1: (python float), score to select model on, higher is better

        """
        # progbar stuff for logging
        batch_size = self.config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size
        #prog = Progbar(target=nbatches)

        # iterate over dataset
        for i, (words, labels) in enumerate(minibatches(train, batch_size)):
            #print(words, labels)
            fd, _ = self.get_feed_dict(words, labels, self.config.lr,
                    self.config.dropout)

            _, train_loss, summary = self.sess.run(
                    [self.train_op, self.loss, self.merged], feed_dict=fd)

            #prog.update(i + 1, [("train loss", train_loss)])

            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch*nbatches + i)

        metrics = self.run_evaluate(dev)
        msg = "Accuracy " + str(metrics["acc"]) + " - F1 " + str(metrics["f1"])
        #msg = " - ".join(["{} {:04.2f}".format(k, v)
        #        for k, v in metrics.items()])
        print(msg)
        self.logger.info(msg)

        return metrics["f1"]


    def run_evaluate(self, test, mode="train"):
        """Evaluates performance on test set

        Args:
            test: dataset that yields tuple of (sentences, tags)

        Returns:
            metrics: (dict) metrics["acc"] = 98.4, ...

        """
        accs = []
        l = []
        #correct_preds_ne, total_correct_ne, total_preds_ne = 0.,0.,0.

        s= ""
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for words, labels in minibatches(test, self.config.batch_size):
            #print(words,labels)
            labels_pred, sequence_lengths, prob = self.predict_batch(words)
            #pdb.set_trace()
            #l.append((list(words),prob)) #list of words, list of scores corresponding
            #l += prob
            #print('labels_pred ', labels_pred)
            if 'test' in mode:
                for lab, pred in zip(labels, labels_pred):
                    #print('lab',lab)
                    #print('pred',pred)
                    for i,j in zip(lab,pred):
                        s+=self.idx_to_tag[i] + '\t' + self.idx_to_tag[j] + '\n'
                    s+='\n'
            for lab, lab_pred, length in zip(labels, labels_pred,
                                             sequence_lengths):
                lab      = lab[:length]
                lab_pred = lab_pred[:length]
                accs    += [a==b for (a, b) in zip(lab, lab_pred)]
                lab_chunks      = set(get_chunks(lab, self.config.vocab_tags))
                lab_pred_chunks = set(get_chunks(lab_pred,
                                                 self.config.vocab_tags))

                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds   += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        #print("Total Preds ", total_preds)
        #print("Total correct ", total_correct)
        #print("Correct preds ", correct_preds)
                
        p   = correct_preds / total_preds if correct_preds > 0 else 0
        r   = correct_preds / total_correct if correct_preds > 0 else 0
        f1  = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)
        if "test" in mode:
            f = open(self.config.file_out + "_" + mode,'w')
            f.write(s)
            f.close()
        #Sort l to get most/least uncertain
        #l2 = sorted(l)
        #mu = []
        #lu = []
        #for i in range(0,self.config.num_query):
        #    mu.append(l.index(l2[i]))
        #    lu.append(l.index(l2[len(l2)-i-1]))
            
        #l = sorted(l, key=lambda pr: pr[2])            
        #pdb.set_trace()
        #print("l",l)
        #return acc, f1, list of most uncertainty and list of least uncertainty examples
        #return {"acc": 100*acc, "f1": 100*f1, "out":l}
        return {"acc": 100*acc, "f1": 100*f1}
        #return {"acc": 100*acc, "f1": 100*f1, "mu": l[0:self.config.num_query], "lu": l[len(l)-self.config.num_query: len(l)]}


    def predict(self, words_raw):
        """Returns list of tags

        Args:
            words_raw: list of words (string), just one sentence (no batch)

        Returns:
            preds: list of tags (string), one for each word in the sentence

        """
        #words = [self.config.processing_word(w) for w in words_raw] #this is used for word raw
        #print(words)
        words = words_raw
        words_o = list(words)
        #print(words_o)
        if type(words[0]) == tuple:
            words = zip(*words)
        #print(words)
        pred_ids, _, scores = self.predict_batch([words])
        #print("Prediction: ")
        #print(pred_ids, _, scores)
        preds = [self.idx_to_tag[idx] for idx in list(pred_ids[0])]
        return (words_o, scores)
        #return preds

def active_strategy(score, transition_params, active_strategy, tag_to_idx):
    """
    Args: output of CRF
    score: A [seq_len, num_tags] matrix of unary potentials.
    transition_params: A [num_tags, num_tags] matrix of binary potentials.
    """
    if active_strategy=="cluster":
        return score
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

    score_final = np.max(trellis[-1]) #Score of sequences (higher = better)
    if (active_strategy=='mg'):
        top_scores = trellis[-1][np.argsort(trellis[-1])[-2:]]
        margin = abs(top_scores[0]-top_scores[1])
        score_final = margin

    elif (active_strategy=='ne'):
        #print("Calling ne strategy")
        #print("score", score)
        #tag_to_idx = {tag: indx for tag, indx in self.config.vocab_tags.items()}
        ne = ['NE.AMBIG','NE.DE', 'NE.LANG3', 'NE.MIXED', 'NE.OTHER','NE.TR']
        ne_idx = []
        for i in tag_to_idx:
            if i in ne:
                ne_idx.append(tag_to_idx[i])
        #print('ne_idx ', ne_idx) 
        #print('score ', score)
        #Get the highest score of NE
        max_ne = []
        #for i in ne_idx:
        #     max_ne.append(np.max(score[:,i]))
        score_final = 0
        for i in viterbi:
            if i in ne_idx:
                score_final+=1 #give higher score to sequences that have more named entities
        #score_final = np.max(max_ne)
    
    elif (active_strategy=='nemg'): #ne margin
        ne_idx = tag_to_idx['NE.DE']
        ne_de = tag_to_idx['DE']
        margin = np.add(score[:,ne_idx],score[:,ne_de])
        margin2 = abs(np.multiply(score[:,ne_idx],score[:,ne_de]))
        margin = np.divide(margin, margin2)
        sum_margin = np.sum(margin)
        score_final = sum_margin

    if (active_strategy=='entropy'):
        #Find the highest prob for each token
        ntoken = len(score)
        ntags = len(score[0])
        l = [] #max prob of each token
        for i in range(0,ntoken):
            l.append(np.max(score[i]))
        ne_idx = tag_to_idx
        #Compute entropy
        score_final = 0.0
        for i in range(0,ntoken):
            score_final+=l[i]*np.log(l[i])
        score_final = score_final/ntoken

    return score_final

def sigmoid(x):
    return 1 / (1 + math.exp(-x))
sigmoid_v = np.vectorize(sigmoid)
