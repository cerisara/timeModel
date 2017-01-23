# -*- coding: utf-8 -*-

from keras.layers import Input, Dense, LSTM, RepeatVector, merge, TimeDistributed, Dropout, Reshape, SimpleRNN, Flatten
from keras.models import Model
from keras.layers.embeddings import Embedding
import numpy as np
from datalib import *
import sys
import codecs

useW2V = False
useRNN = False
useWordLen = True

dimrnn=5
winlen=5
nepochs=50
dimpos=10
if len(sys.argv)>1: dimrnn=int(sys.argv[1])

''' train un modele en estimant d'abord par cross-val le nb d'epoch optimum:
    - cross-val à 0.3 puis à 0.15, puis on extrapole lineairement pour avoir le nb d'epoch à 0
'''
def trainOptimumEpoch(model,datax, datay, nepochsmax):
    w0=model.get_weights()
    val=0.3
    bestloss,bestepoch0=np.inf,-1
    for epoch in range(nepochsmax):
        h=model.fit(datax,datay,nb_epoch=1,shuffle=True,verbose=2,validation_split=val)
        # print("loss "+str(h.history['loss'][0])+" "+str(h.history['val_loss'][0]))
        if h.history['val_loss'][0]<bestloss:
            bestloss=h.history['val_loss'][0]
            bestepoch0=epoch
    model.set_weights(w0)
    val=0.15
    bestloss,bestepoch1=np.inf,-1
    for epoch in range(nepochsmax):
        h=model.fit(datax,datay,nb_epoch=1,shuffle=True,verbose=2,validation_split=val)
        # print("loss "+str(h.history['loss'][0])+" "+str(h.history['val_loss'][0]))
        if h.history['val_loss'][0]<bestloss:
            bestloss=h.history['val_loss'][0]
            bestepoch1=epoch
    # coef mult
    a=float(bestepoch1)/float(bestepoch0)
    print("epochcoef "+str(a))
    if a<1.: a=1. # interdit de diminuer le nb d epochs
    elif a>1.5: a=1.5 # on n'augmente pas trop non plus
    nep = int(float(bestepoch0)*a*a)
    model.set_weights(w0)
    h=model.fit(datax,datay,nb_epoch=nep,shuffle=True,verbose=2)
    return h

datatrain=Corpus()
datatrain.loadTrain()
trainwords=datatrain.getWords()
# for w in trainwords: print("DETWORD "+w)

emb=WordEmb()

vocpos = Voc()
vocpos.setWords(datatrain.getPOS())

# read temporal embeddings
#mywemb = {}
#mydim=-1
#with codecs.open('../gigaword/wembeds.txt','rb','utf-8') as f:
#    while True:
#        l=f.readline()
#        if not l: break
#        ll=l.strip().split()
#        w=ll[0]
#        e=np.zeros((len(ll)-1,),dtype='float32')
#        for i in range(1,len(ll)): e[i-1]=float(ll[i])
#        mywemb[w]=e
#        if mydim<0: mydim=len(e)
#        else: assert len(e)==mydim
#print("my embeddings read",len(mywemb),mydim)

print("build model",vocpos.len())
inWembed=Input(shape=(emb.getNdims(),),dtype='float32')
inPOS=Input(shape=(winlen,),dtype='int32')
inWlen=Input(shape=(1,),dtype='float32')
posemb=Embedding(vocpos.len(),dimpos,input_length=winlen)(inPOS)
getin=[]
inputs=[]
if useW2V:
	getin.append(inWembed)
	inputs.append(inWembed)
inputs.append(inPOS)
if useRNN:
    h=SimpleRNN(dimrnn)(posemb)
    getin.append(h)
else: getin.append(Flatten()(posemb))
if useWordLen:
	getin.append(inWlen)
	inputs.append(inWlen)
if len(getin)==1: h=getin
else: h=merge(getin,mode='concat')

x=Dense(2,activation='softmax')(h)
model=Model(input=inputs,output=[x])
model.compile('adam', loss=['categorical_crossentropy'], metrics=['accuracy'])
print(model.summary())

print("prepare data arrays",len(trainwords))
ax=np.zeros((len(trainwords),emb.getNdims()),dtype=np.float32)
axx=np.zeros((len(trainwords),winlen),dtype=np.int32)
axl=np.zeros((len(trainwords),1),dtype=np.float32)
ay=np.zeros((len(trainwords),2),dtype=np.float32)
for i in range(len(trainwords)):
    # la methode vect() convertit en lowercase en interne
    w=trainwords[i]
    ax[i][:]=emb.vect(w)[:]
    axl[i][0]=float(len(w))/10.
    for k in range(winlen):
        j=k-winlen//2
        postag='PAD'
        if i+j>=0 and i+j<len(trainwords): postag=datatrain.getPOS()[i+j]
        axx[i][k]=vocpos.getWordIdx(postag)
    ay[i][datatrain.getLabel(i)]=1

print("train")
datins = []
if useW2V: datins.append(ax)
datins.append(axx)
if useWordLen: datins.append(axl)
h=trainOptimumEpoch(model,datins,[ay], nepochs)

print("test")
datatest=Corpus()
datatest.loadTest()
testwords=datatest.getWords()

ax=np.zeros((len(testwords),emb.getNdims()),dtype=np.float32)
axx=np.zeros((len(testwords),winlen),dtype=np.int32)
axl=np.zeros((len(testwords),1),dtype=np.float32)
ay=[0]*len(testwords)
for i in range(len(testwords)):
    w=testwords[i]
    # la methode vect() convertit en lowercase en interne
    ax[i][:]=emb.vect(w)[:]
    axl[i][0]=float(len(w))/10.
    for k in range(winlen):
        j=k-winlen//2
        postag='PAD'
        if i+j>=0 and i+j<len(testwords): postag=datatest.getPOS()[i+j]
        axx[i][k]=vocpos.getWordIdx(postag)
    ay[i]=datatest.getLabel(i)
datins=[]
if useW2V: datins.append(ax)
datins.append(axx)
if useWordLen: datins.append(axl)
rec=model.predict(datins)

# datatest.curves(rec,ay)
# l appel de cette fct va cree rec.tab, qui sera utilise dans le script d'eval de tempeval2
datatest.testeval(rec)
datatest.printErrors(rec,ay)

