## buildedu.py
## Author: Yangfeng Ji
## Date: 05-03-2015
## Time-stamp: <yangfeng 09/25/2015 15:35:08>

from os import listdir
from os.path import join, basename
from model.classifier import Classifier
from model.docreader import DocReader
from model.sample import SampleGenerator
from cPickle import load
import gzip
import re
import numpy as np


def main(fmodel, fvocab, rpath, wpath, edu_path =True):
    clf = Classifier()
    dr = DocReader()
    clf.loadmodel(fmodel)
    flist = [join(rpath,fname) for fname in listdir(rpath) if fname.endswith('conll')]
    # flist = flist[115:]
    vocab = load(gzip.open(fvocab))
    for (fidx, fname) in enumerate(flist):
        print "Processing file: {}".format(fname)
        doc = dr.read(fname, withboundary=False)
        sg = SampleGenerator(vocab)
        sg.build(doc)

        M, _ = sg.getmat()
        
        ## use true EDU files
        if edu_path:
            edup = fname.replace('.conll', '.edus')
            predlabels = get_true_labels(edup,fname) 
        else:
            ## use classifier
            predlabels = clf.predict(M)
        
        doc = postprocess(doc, predlabels)
        eduidx = writedoc(doc, fname, wpath)
        check(eduidx,edup)


def postprocess(doc, predlabels):
    """ Assign predlabels into doc
    """
    tokendict = doc.tokendict
    for gidx in tokendict.iterkeys():
        if predlabels[gidx] == 1:
            tokendict[gidx].boundary = True
        else:
            tokendict[gidx].boundary = False
        if tokendict[gidx].send:
            tokendict[gidx].boundary = True
    return doc


# def writedoc(doc, fname, wpath):
#     """ Write doc into a file with the CoNLL-like format
#     """
#     tokendict = doc.tokendict
#     N = len(tokendict)
#     fname = basename(fname) + '.edu'
#     fname = join(wpath, fname)
#     eduidx = 0
#     with open(fname, 'w') as fout:
#         for gidx in range(N):
#             fout.write(str(eduidx) + '\n')
#             if tokendict[gidx].boundary:
#                 eduidx += 1
#             if tokendict[gidx].send:
#                 fout.write('\n')
#     print 'Write segmentation: {}'.format(fname)


def writedoc(doc, fname, wpath):
    """ Write file
    """
    tokendict = doc.tokendict
    N = len(tokendict)
    fname = basename(fname).replace(".conll", ".merge")
    fname = join(wpath, fname)
    eduidx = 1
    with open(fname, 'w') as fout:
        for gidx in range(N):
            tok = tokendict[gidx]
            line = str(tok.sidx) + "\t" + str(tok.tidx) + "\t"
            line += tok.word + "\t" + tok.lemma + "\t" 
            line += tok.pos + "\t" + tok.deplabel + "\t" 
            line += str(tok.hidx) + "\t" + tok.ner + "\t"
            line += tok.partialparse + "\t" + str(eduidx) + "\n"
            fout.write(line)
            # Boundary
            if tok.boundary:
                eduidx += 1
            if tok.send:
                fout.write("\n")
    return eduidx


def get_true_labels(edup, conllp):
    """return a list of indicating boundaries"""
    try:
        edus = open(edup).readlines()
        conll = open(conllp).readlines()
    except IOError:
        print("No such file",edup)
        return
    edus = [a.strip("\n") for a in edus]
    edus = [a.strip() for a in edus]
    
    conll = [c.strip("\n") for c in conll]
    orig = [c.split("\t")[2] for c in conll if c]
    bd = []
    count = 0
    idx = 0
    # print(edus)
    # print(orig)
    for w in orig:
        # print(w,edus[idx])
        escaped = False
        matchidx = np.array([m.start() for m in re.finditer(re.escape(w), edus[idx])])
        if re.escape(w) != w:
            escaped = True
        # print(w,edus[idx],count,matchidx)
        if (matchidx >= count).any():
            count += len(w) if escaped else len(w)+1
            bd.append(0)
        else:
            idx += 1
            count = len(w) if escaped else len(w)+1
            bd[-1]=1
            bd.append(0)
    bd[-1]=1
    bd = np.array(bd)
    assert np.sum(bd==1)==len(edus),bd
    return bd

def check(eduidx, edup):
    """check if #edu in .edu is the same as eduidx"""
    try:
        edus = open(edup).readlines()
    except IOError:
        print("No such file",edup)
        return
    edus = [a.strip("\n") for a in edus]
    edus = [a.strip() for a in edus]
    if len(edus)!=eduidx-1:
        print(edup)
    # assert len(edus)==eduidx-1,(eduidx,edup)
