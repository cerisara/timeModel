# -*- coding: utf-8 -*-

# TODO: j ai modifie un duplicat de cette classe dans ../gigaword/voclib.py qui ne contient pas les bilios graphiques
#       il faut donc supprimer les redondances ici et l'importer

from collections import Counter
from scipy import spatial
import numpy as np
import word2vec
import sys
import codecs
import os
# import bob.measure
# from matplotlib import pyplot

reload(sys) # just to be sure
sys.setdefaultencoding('utf-8')

''' construit un mapping mot -> index '''
class Voc:
    def __init__(self):
        self.voc=Counter()

    def addWord(self,w):
        i=len(self.voc)+1
        self.voc[w]=i

    def setWords(self,ws):
        self.voc=Counter({x:i+1 for i,x in enumerate(set(ws))})
        self.vocinv = {v: k for k, v in self.voc.items()}

    ''' return 0 if unknown '''
    def getWordIdx(self,w):
        return self.voc[w]

    def getWordStr(self, widx):
        return self.vocinv[widx]

    def len(self):
        return len(self.voc)+1 # for unknown

    def save(self,fich):
        with open(fich,'w') as f:
            for w,i in self.voc.items(): f.write(w+"\t"+str(i)+'\n')

    def load(self,fich):
        with open(fich,'r') as f:
            lines=f.readlines()
            for l in lines:
                s=l.strip().split()
                self.voc[s[0]]=int(s[1])
        self.vocinv = {v: k for k, v in self.voc.items()}

    def load(self,fich,n):
        with open(fich,'r') as f:
            lines=f.readlines()
            for l in lines[0:min(len(lines),n)]:
                s=l.strip().split()
                self.voc[s[0]]=int(s[1])
        self.vocinv = {v: k for k, v in self.voc.items()}

    ''' ces 2 methodes permettent de compter les mots; attention: elles ne construisent pas le mapping vers les index ! '''
    def countWords(self, ws):
        self.voc.update(ws)

    def getMostFrequent(self, n):
        mostCommon = self.voc.most_common(n)
        return mostCommon

''' classe qui donne acces aux pretrained words embeddings
    Exemple d'utilisation:
        w2v=WordEmb()
        print(w2v.dist(u'défi',u'problème'))
        print(w2v.dist(u'défi',u'artichaud'))
'''
class WordEmb:
    def __init__(self):
        from os.path import expanduser
        home = expanduser("~")
        print("loading words embeddings")
        self.model = word2vec.load(home+'/frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin')
        self.unk = [0]*200
        # test accents
        v=self.model.get_vector(u'à')
        print("loading done",v,type(u'à'))

    def dist(self,wa,wb):
        if wa not in self.model: return 1
        va=self.model.get_vector(wa)
        if wb not in self.model: return 1
        vb=self.model.get_vector(wb)
        d=spatial.distance.cosine(va,vb)
        return d

    def closest(self,wa):
        indexes, scores = self.model.cosine(wa)
        res=self.model.generate_response(indexes,scores).tolist()
        return res

    def vect(self,wa):
        wa=wa.lower()
        wa=wa.decode('utf8')
        if wa not in self.model:
            # print("WARNING unknown "+wa)
            return self.unk
        va=self.model.get_vector(wa)
        return va

    def getNdims(self):
        return 200

class Corpus:
    def __init__(self):
        self.testfichs=("2003-01-15-0002.xml","2003-01-15-0005.xml","2003-01-15-0008.xml","algerie.xml","baldwin_frratrain_15.xml","bio_butler.xml","bove.xml","RDS020607SENATEURS.xml","texte5.xml")

    def loadTab(self,ff,debug=False):
        with open(ff) as f: lines=f.readlines()
        mots,pos,lem,tag=[],[],[],[]
        for l in lines:
            s=l.split()
            if len(s)>=4:
                mots.append(s[0])
                pos.append(s[1])
                lem.append(s[2])
                tag.append(s[3])
        if debug: print("loadtab nlines",len(lines),len(mots))
        return mots,pos,lem,tag

    def loadTrain(self):
        fichs=os.listdir("data/train/")
        data=[[],[],[],[]]
        for tabfile in fichs:
            if not tabfile.endswith('.xml.tab'): continue
            mots,pos,lem,tag = self.loadTab("data/train/"+tabfile)
            data[0]+=mots
            data[1]+=pos
            data[2]+=lem
            data[3]+=tag
        self.data=data

    def loadTest(self):
        data=[[],[],[],[]]
        for tabfile in self.testfichs:
            mots,pos,lem,tag = self.loadTab("data/test/"+tabfile+'.tab')
            data[0]+=mots
            data[1]+=pos
            data[2]+=lem
            data[3]+=tag
        self.data=data

    def printErrors(self,modeloutput,labels):
#        n1=sum(labels)
#        positives=np.zeros((n1,),dtype=np.float64)
#        negatives=np.zeros((len(labels)-n1,),dtype=np.float64)
#        un,up=0,0
#        for i in range(len(labels)):
#            if labels[i]==0:
#                negatives[un]=modeloutput[i][1]
#                un+=1
#            else:
#                positives[up]=modeloutput[i][1]
#                up+=1
#        thr=0.5
#        prec,rec = bob.measure.precision_recall(negatives,positives,thr)
#        f1=2.*prec*rec/(prec+rec)
#        print("myprecrec",thr,prec,rec,f1)
        
        for i in range(len(labels)):
            if labels[i]==1: gp=1
            else: gp=0
            if modeloutput[i][1]>0.5: rp=1
            else: rp=0
            if gp!=rp:
                s=""
                for j in range(max(0,i-4),min(len(labels),i+4)): s+=self.getWords()[j]+' '
                print("ERROR\t"+str(gp)+"\t"+str(modeloutput[i][1])+"\t"+self.getPOS()[i]+'\t'+self.getWords()[i]+'\t'+s)

#    def curves(self,modeloutput,labels):
#        n1=sum(labels)
#        positives=np.zeros((n1,),dtype=np.float64)
#        negatives=np.zeros((len(labels)-n1,),dtype=np.float64)
#        un,up=0,0
#        for i in range(len(labels)):
#            if labels[i]==0:
#                negatives[un]=modeloutput[i][1]
#                un+=1
#            else:
#                positives[up]=modeloutput[i][1]
#                up+=1
#        bob.measure.plot.precision_recall_curve(negatives, positives, 100, color=(0,0,0), linestyle='-', label='MLP')
#        pyplot.savefig('pr.pdf')
#        prec,rec = bob.measure.precision_recall(negatives,positives,0.5)
#        f1=2.*prec*rec/(prec+rec)
#        print("myprecrec",prec,rec,f1)

    def testeval(self,modeloutput):
        # just to check that the nb of words per file is the same in testbase.tab and in the original XML file
        with open("testbase.tab") as f:
            lines=[l.split()[0] for l in f.readlines()]
            cobase=Counter(lines)
        modidx=0
        with open('rec.tab','w') as recf:
            for xmlfile in self.testfichs:
                tabfile=xmlfile+'.tab'
                mots,pos,lem,tag = self.loadTab(tabfile)
                nex=len(mots)
                print(xmlfile,nex,cobase[xmlfile])
                assert nex==cobase[xmlfile]
                rec=modeloutput[modidx:modidx+nex]
                modidx+=nex
                nev=0
                for i in range(len(rec)):
                    if np.argmax(rec[i])==1:
                        nev+=1
                        recf.write(xmlfile+'\t'+'0\t'+str(i)+'\t'+'timex3\t'+'t'+str(nev)+'\t'+'1\t'+'type\t'+'EVENT\n')
        assert modidx==len(modeloutput)

    def getWords(self):
        return self.data[0]

    def getPOS(self):
        return self.data[1]

    def getLabel(self,i):
        if self.data[3][i]=='O': return 0
        else: return 1

class Conll:
    def __init__(self,fich):
        self.f=codecs.open(fich,'r','utf-8')
        self.isopen=True

    def nextSent(self):
        if not self.isopen: return None
        words=[]
        while True:
            l=self.f.readline()
            if not l:
                self.f.close()
                self.isopen=False
                return None
            l=l.strip()
            if len(l)==0: break
            words.append(l.split('\t')[1])
        return words


