## model.py
## Author: Yangfeng Ji
## Date: 09-09-2014
## Time-stamp: <yangfeng 09/27/2015 12:32:37>

""" As a parsing model, it includes the following functions
1, Mini-batch training on the data generated by the Data class
2, Shift-Reduce RST parsing for a given text sequence
3, Save/load parsing model
"""
# from keras.utils import to_categorical
# from keras.models import Sequential
# from keras.layers import *
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from pickle import load, dump
from .parser import SRParser
from .feature import FeatureGenerator
from .tree import RSTTree
from .util import *
from .datastructure import ActionError
from operator import itemgetter
import gzip, sys

class ParsingModel(object):
    def __init__(self, vocab=None, idxlabelmap=None, clf=None,
                 withdp=None, fdpvocab=None, fprojmat=None, n_svd=0):
        """ Initialization
        
        :type vocab:
        :param vocab:

        :type idxrelamap:
        :param idxrelamap:

        :type clf:
        :param clf:
        """
        self.vocab = vocab
        # print labelmap
        self.labelmap = idxlabelmap
        if clf is None:
            self.clf = LinearSVC(C=1.0, penalty='l1',
                loss='squared_hinge', dual=False, tol=1e-7)
            # self.clf = LogisticRegression(C=1.0,penalty="l1",tol=1e-7)
            # self.clf = build_nn()
        else:
            self.clf = clf
        self.withdp = withdp
        self.dpvocab, self.projmat = None, None
        self.n_svd = n_svd
        self.svd = None
        if self.n_svd > 0:
            self.svd = TruncatedSVD(n_components=n_svd, n_iter=7, random_state=42)
        if withdp:
            print('Loading projection matrix ...')
            with gzip.open(fdpvocab) as fin:
                self.dpvocab = load(fin)
            with gzip.open(fprojmat) as fin:
                self.projmat = load(fin)
        print('Finish initializing ParsingModel')


    def train(self, trnM, trnL):
        """ Perform batch-learning on parsing model
        """
        print('Training ...')
        if self.svd:
            trnM = self.svd.fit_transform(trnM)
            # print("trnM shape",trnM.shape)
        # trnL = to_categorical(trnL)
        # self.clf.fit(trnM.toarray(), trnL,validation_split=0.1,epochs=3)
        self.clf.fit(trnM, trnL)

    def predict(self, features):
        """ Predict parsing actions for a given set
            of features

        :type features: list
        :param features: feature list generated by
                         FeatureGenerator
        """
        vec = vectorize(features, self.vocab,
                        self.dpvocab, self.projmat)
        if self.svd:
            vec = self.svd.transform(vec)
        label = self.clf.predict(vec)
        
        return self.labelmap[label[0]]


    def rank_labels(self, features):
        """ Rank the decision label with their confidence
            value
        """
        vec = vectorize(features, self.vocab,
                        self.dpvocab, self.projmat)
        
        if self.svd:
            vec = self.svd.transform(vec)
        # vals = self.clf.predict(vec.toarray())
        vals = self.clf.decision_function(vec)
        
        labelvals = {}
        for idx in range(len(self.labelmap)):
            labelvals[self.labelmap[idx]] = vals[0,idx]
        sortedlabels = sorted(list(labelvals.items()), key=itemgetter(1),
                              reverse=True)
        labels = [item[0] for item in sortedlabels]
        return labels


    def savemodel(self, fname):
        """ Save model and vocab
        """
        if not fname.endswith('.gz'):
            fname += '.gz'
        D = {'clf':self.clf, 'vocab':self.vocab,
             'idxlabelmap':self.labelmap,"svd":self.svd}
        with gzip.open(fname, 'w') as fout:
            dump(D, fout)
        print('Save model into file: {}'.format(fname))


    def loadmodel(self, fname):
        """ Load model
        """
        with gzip.open(fname, 'r') as fin:
            D = load(fin)
        self.clf = D['clf']
        self.vocab = D['vocab']
        self.labelmap = D['idxlabelmap']
        self.svd = D['svd']
        print('Load model from file: {}'.format(fname))


    def sr_parse(self, doc, bcvocab=None):
        """ Shift-reduce RST parsing based on model prediction

        :type texts: list of string
        :param texts: list of EDUs for parsing

        :type bcvocab: dict
        :param bcvocab: brown clusters
        """
        # raise NotImplementedError("Not finished yet")
        # Initialize parser
        srparser = SRParser([],[])
        srparser.init(doc)
        # Parsing
        while not srparser.endparsing():
            # Generate features
            stack, queue = srparser.getstatus()
            # Make sure call the generator with
            # same arguments as in data generation part
            fg = FeatureGenerator(stack, queue, doc, bcvocab)
            feat = fg.features()
            # label = self.predict(feat)
            labels = self.rank_labels(feat)
            for label in labels:
                action = label2action(label)
                try:
                    srparser.operate(action)
                    break
                except ActionError:
                    # print "Parsing action error with {}".format(action)
                    pass
        tree = srparser.getparsetree()
        rst = RSTTree()
        rst.asign_tree(tree)
        return rst
    

def build_nn():
    model = Sequential()
    model.add(Dense(80, input_dim=8000,activation="tanh"))
    # model.add(Dense(60,activation="tanh"))
    model.add(Dense(4,activation="softmax"))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
    return model
            
