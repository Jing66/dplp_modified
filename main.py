## main.py
## Author: Yangfeng Ji
## Date: 02-14-2015
## Time-stamp: <yangfeng 09/25/2015 22:03:48>

from code.readdoc import readdoc
from code.data import Data
from code.model import ParsingModel
from code.util import reversedict
from code.evalparser import evalparser
from pickle import load,dump
import gzip
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.sparse.coo import coo_matrix
from scipy.linalg import norm

WITHDP = True
N_SVD = 0
K = 150
TOL = 0.1


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
    # word_dict = dict([(v,k[-1]) for k,v in data.vocab.items()])
    # with gzip.open("data/resources/word-dict.pickle.gz", 'w') as fout:
    #     dump(word_dict, fout)
    # print(len(data.vocab),len(word_dict))
   

def trainmodel():
    fvocab = "data/sample/vocab.pickle.gz"
    fdata = "data/sample/trn.data"
    flabel = "data/sample/trn.label"
    D = load(gzip.open(fvocab))
    vocab, labelidxmap = D['vocab'], D['labelidxmap']
    print('len(vocab) = {}'.format(len(vocab)))
    data = Data()
    trnM, trnL = data.loadmatrix(fdata, flabel)
    print('trnM.shape = {}'.format(trnM.shape))
    idxlabelmap = reversedict(labelidxmap)
    pm = ParsingModel(vocab=vocab, idxlabelmap=idxlabelmap,
                withdp = WITHDP,
                fdpvocab="data/resources/word-dict.pickle.gz",
               fprojmat="data/resources/projmat.pickle.gz",
               n_svd=N_SVD)
    pm.train(trnM, trnL)
    # pm.savemodel("model/parsing-model.pickle.gz")
    evalparser(path="data/test/", report=True, 
               bcvocab=bcvocab, draw=False,
               withdp=WITHDP,
               fdpvocab="data/resources/word-dict.pickle.gz",
               fprojmat="data/resources/projmat.pickle.gz",pm=pm)


def train_projm(T = 500):
    fdata = "data/sample/trn.data"
    flabel = "data/sample/trn.label"
    data = Data()
    trnM, trnL = data.loadmatrix(fdata, flabel)
    trnM = trnM.toarray()
    print('trnM.shape = ',trnM.shape)
    print("n class",len(set(trnL)))
    epsilon = 1e-2
    batch_size = 500
    A_prev = np.random.rand(trnM.shape[1],K)
    mask0 = np.arange(3)
    mask1 = np.array([0,3,4])
    mask2 = np.array([1,3,5])
    mask3 = np.array([2,4,5])
    scaler = StandardScaler()
    A1 = None
    A2 = None
    for t in range(1,T):
        print(("iteration",t))
        p_ = np.random.choice(trnM.shape[0],batch_size,replace=False).astype(int)       
        trnM_t, trnL_t = trnM[p_], np.array(trnL)[p_]
        n_class = set(trnL_t)
        # print("n_class",n_class)
        trnM_t = trnM_t.dot(A_prev)
        trnM_t = scaler.fit_transform(trnM_t)
        clf = SVC(C=1.0, kernel="linear", tol=TOL)
        clf.fit(trnM_t, trnL_t)
        pweight = clf.coef_
        dweight = clf.dual_coef_
        # print("primal weight shape",pweight.shape,"dual weight shape",dweight.shape)
        A_new = (1-TOL/t)*A_prev
        s = 0
        print("Updating A...")
        for i,svid in enumerate(clf.support_):
            yi = trnL_t[svid]
            if yi==0:
                mask = mask0
                masks = np.concatenate((mask1,mask2,mask3))
            elif yi==1:
                mask = mask1
                masks = np.concatenate((mask0,mask2,mask3))
            elif yi==2:
                mask = mask2
                masks = np.concatenate((mask0,mask1,mask3))
            else:
                mask = mask3
                masks = np.concatenate((mask0,mask1,mask2))
            masks = masks.reshape((3,3))
            pweight_ = pweight[mask,:]
            # print("pweight_ shape",pweight_.shape, "dweight shape",dweight.shape)
            # print(pweight[masks].T.shape)
            inner = pweight_ - dweight[:,i].T.dot(pweight[masks].T.T)
            # print("inner.shape",inner.shape) # (n_class-1,K)
            s += 1/t * np.sum(inner.dot(trnM_t[svid].T))
        A_new += s
        if t==1:
            A1 = A_new
        elif t==2:
            A2 = A_new
        if A1 is not None and A2 is not None:
            d = norm(A_new - A_prev)/norm(A2-A1)
            print("Distance",d)
            if t>2 and d< epsilon:
                print(("stop iteration",t))
                with gzip.open("data/resources/projmat.pickle.gz", 'w') as fout:
                    dump(A_new, fout)
                exit(0)
        A_prev = A_new
    
    
     
    


if __name__ == '__main__':
    bcvocab=None
    ## Use brown clsuters
    with gzip.open("resources/bc3200.pickle.gz") as fin:
       
        bcvocab = load(fin)
    # Create training data
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
