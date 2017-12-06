## main.py
## Author: Yangfeng Ji
## Date: 02-14-2015
## Time-stamp: <yangfeng 09/25/2015 22:03:48>

from code.readdoc import readdoc
from code.data import Data
from code.model import ParsingModel
from code.util import reversedict
from code.evalparser import evalparser
from cPickle import load,dump
import gzip
from sklearn.svm import SVC
import numpy as np
from scipy.sparse.coo import coo_matrix

WITHDP = False
N_SVD = 0


def createdoc():
    ftrn = "data/sample/trn-doc.pickle.gz"
    rpath = "data/training/"
    readdoc(rpath, ftrn)
    ftst = "data/sample/tst-doc.pickle.gz"
    rpath = "data/test/"
    readdoc(rpath, ftst)


def createtrndata(path="data/training/", topn=10000, bcvocab=None):
    data = Data(bcvocab=bcvocab,
                withdp=WITHDP,
                fdpvocab="data/resources/word-dict.pickle.gz",
                fprojmat="data/resources/projmat.pickle.gz")
    data.builddata(path)
    data.buildvocab(topn=topn)
    data.buildmatrix()
    
    fdata = "data/sample/trn.data"
    flabel = "data/sample/trn.label"
    data.savematrix(fdata, flabel)
    data.savevocab("data/sample/vocab.pickle.gz")

    ## save word-dict for later projection
    word_dict = dict([(k[-1],v) for k,v in data.vocab.iteritems()])
    with gzip.open("data/resources/word-dict.pickle.gz", 'w') as fout:
        dump(word_dict, fout)
    

def trainmodel():
    fvocab = "data/sample/vocab.pickle.gz"
    fdata = "data/sample/trn.data"
    flabel = "data/sample/trn.label"
    D = load(gzip.open(fvocab))
    vocab, labelidxmap = D['vocab'], D['labelidxmap']
    print 'len(vocab) = {}'.format(len(vocab))
    data = Data()
    trnM, trnL = data.loadmatrix(fdata, flabel)
    print 'trnM.shape = {}'.format(trnM.shape)
    idxlabelmap = reversedict(labelidxmap)
    pm = ParsingModel(vocab=vocab, idxlabelmap=idxlabelmap, n_svd=N_SVD)
    pm.train(trnM, trnL)
    # pm.savemodel("model/parsing-model.pickle.gz")
    evalparser(path="data/test/", report=True, 
               bcvocab=bcvocab, draw=False,
               withdp=WITHDP,
               fdpvocab="data/resources/word-dict.pickle.gz",
               fprojmat="data/resources/projmat.pickle.gz",pm=pm)

def train_projm(T = 50):
    fdata = "data/sample/trn.data"
    flabel = "data/sample/trn.label"
    data = Data()
    trnM, trnL = data.loadmatrix(fdata, flabel)
    trnM = trnM.toarray()
    clf = SVC(C=1.0, kernel="linear", tol=1e-7)
    print 'trnM.shape = {}'.format(trnM.shape)
    epsilon = 1e-4
    K = 60
    batch_size = 500
    tau = 0.01
    A_prev = np.random.rand(trnM.shape[1],K)
    for t in range(1,T):
        print("iteration",t)
        p_ = np.random.choice(trnM.shape[0],batch_size,replace=False).astype(int)       
        trnM_t, trnL_t = trnM[p_], np.array(trnL)[p_]
        n_class = set(trnL_t)
        # print("n_class",n_class)
        trnM_t = trnM_t.dot(A_prev)
        clf.fit(trnM_t, trnL_t)
        pweight = clf.coef_
        dweight = clf.dual_coef_
        # print("primal weight shape",pweight.shape,"dual weight shape",dweight.shape)
        A_new = (1-tau/t)*A_prev
        s = 0
        for i,svid in enumerate(clf.support_):
            yi = trnL_t[svid]
            if yi==0:
                mask = np.arange(3)
            elif yi==1:
                mask = np.array([0,3,4])   
            elif yi==2:
                mask = np.array([1,3,5])
            else:
                mask = np.array([2,4,5])
            pweight_ = pweight[mask,:]
            # print("pweight_ shape",pweight_.shape, mask)
            # print(dweight[:,i].shape, pweight_[svid].shape)
            inner = pweight_ - dweight[:,i].dot(pweight_)
            # print("inner.shape",inner.shape)
            s+= 1/t * np.sum(inner.dot(trnM_t[svid].T))
        A_new += s
        if t>2 and (np.sqrt(np.sum(np.square(A_new-A_prev)))< epsilon).all():
            print("stop iteration",t)
            break
        A_prev = A_new
    
    with gzip.open("data/resources/projmat.pickle.gz", 'w') as fout:
        dump(A_new, fout)
     
    


if __name__ == '__main__':
    bcvocab=None
    ## Use brown clsuters
    # with gzip.open("resources/bc3200.pickle.gz") as fin:
    #     print 'Load Brown clusters for creating features ...'
    #     bcvocab = load(fin)
    ## Create training data
    # createtrndata(path="data/training/", topn=8000, bcvocab=bcvocab)

    ## Train model and evaluate
    trainmodel()
    # train_projm()

    ## Evaluate model on the RST-DT test set
    # evalparser(path="data/test/", report=True, 
            #    bcvocab=bcvocab, draw=False,
            #    withdp=WITHDP,
            #    fdpvocab="data/resources/word-dict.pickle.gz",
            #    fprojmat="data/resources/projmat.pickle.gz")
