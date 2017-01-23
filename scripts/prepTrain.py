# -*- coding: utf-8 -*-

import sys
import os
import re
from alignment.sequence import Sequence
from alignment.vocabulary import Vocabulary
from alignment.sequencealigner import SimpleScoring, GlobalSequenceAligner

treetaggerpath = "/home/xtof/softs/treetagger/"

reload(sys) # just to be sure
sys.setdefaultencoding('utf-8')

allfichs = ("1999-05-17-0121.xml","1999-05-18-0333.xml","1999-05-19-0489a.xml","1999-05-19-0543.xml","1999-05-20-0108.xml","1999-05-20-1029.xml","1999-05-21-0100.xml","1999-05-21-0102.xml","1999-05-21-0104.xml","1999-05-22-0991.xml","1999-05-23-0078.xml","1999-05-27-0530a.xml","1999-05-27-0530b.xml","1999-05-27-0530d.xml","1999-05-27-0530e.xml","1999-05-29-0648.xml","1999-06-04-0535b.xml","1999-06-04-0535c.xml","1999-06-12-0548c.xml","1999-06-12-0548d.xml","1999-06-12-1010b.xml","1999-07-09-0441.xml","1999-07-16-0592.xml","1999-07-23-0052.xml","1999-07-24-0101.xml","1999-07-28-0348a.xml","1999-07-28-0348c.xml","1999-07-28-0348d.xml","1999-07-28-0348e.xml","1999-07-28-0438.xml","1999-08-06-0305.xml","1999-08-07-0376c.xml","1999-08-07-0951.xml","1999-08-10-0788.xml","1999-08-27-1122.xml","1999-09-03-0450b.xml","1999-09-04-0945.xml","1999-09-05-0496.xml","1999-09-30-0406.xml","2002-01-03-0106.xml","2002-01-21-0155.xml","2002-02-09-0883.xml","2002-03-27-0154.xml","2002-04-02-0412a.xml","2002-04-03-1077.xml","2002-04-21-0127.xml","2002-04-30-0498c.xml","2002-05-07-0144.xml","2002-08-25-0493b.xml","2002-09-22-0531.xml","2002-09-29-0965.xml","2002-11-04-0354.xml","2002-11-21-0215.xml","2003-01-04-1029.xml","2003-01-15-0002.xml","2003-01-15-0005.xml","2003-01-15-0008.xml","2003-01-15-0009.xml","2003-01-15-0010.xml","2003-01-21-0468.xml","2003-01-24-0478a.xml","2003-01-27-0129.xml","2003-02-04-0026.xml","2003-02-04-0456a.xml","2003-02-04-0456d.xml","2003-02-04-0456e.xml","2003-02-04-0872.xml","2003-02-07-0005.xml","2003-02-21-0862.xml","2003-02-22-0872.xml","2003-02-23-0082.xml","2003-02-23-0086.xml","2003-02-24-0122.xml","algerie.xml","baldwin_frratrain_15.xml","baldwin_frratrain_28.xml","bio_butler.xml","bove.xml","RDS010607FLAMES.xml","RDS020607SENATEURS.xml","texte4.xml","texte5.xml","texte8.xml")

testfichs=("2003-01-15-0002.xml","2003-01-15-0005.xml","2003-01-15-0008.xml","algerie.xml","baldwin_frratrain_15.xml","bio_butler.xml","bove.xml","RDS020607SENATEURS.xml","texte5.xml")

print("lentrain0",len(allfichs))
trainfichs=list(set(allfichs)-set(testfichs))
print("lentrain1",len(trainfichs))

def alignToks(gold,rec):
    r=Sequence(rec)
    g=Sequence(gold)
    v=Vocabulary()
    re=v.encodeSequence(r)
    rg=v.encodeSequence(g)
    scoring=SimpleScoring(2,-1)
    aligner=GlobalSequenceAligner(scoring, -2)
    score, encodeds = aligner.align(re, rg, backtrace=True)

    n1,n2=0,0
    for i in range(len(encodeds[0])):
        if encodeds[0].first[i]>0: n1+=1
        if encodeds[0].second[i]>0: n2+=1
#    print("NNN",n1,n2,len(rec),len(gold))

    print("align score",score,rec[0])
    gi,ri=0,0
    if len(rec)>0:
        for i in range(len(rec)):
            if v.decode(encodeds[0].first[0])==rec[i]: break
            else: ri+=1
    if len(gold)>0:
        for i in range(len(gold)):
            if v.decode(encodeds[0].second[0])==gold[i]: break
            else: gi+=1
    err=0
    indexes=[]
    for i in range(len(encodeds[0])):
#        print("dbug",i,len(encodeds[0]),encodeds[0].first[i],encodeds[0].second[i],v.decode(encodeds[0].first[i]),v.decode(encodeds[0].second[i]),rec[ri],gold[gi])
        if encodeds[0].first[i]>0 and encodeds[0].second[i]>0:
            # gi et ri sont alignes
            if rec[ri].lower() == gold[gi].lower() or rec[ri].lower().startswith(gold[gi].lower()) or gold[gi].lower().startswith(rec[ri].lower()) or rec[ri].lower().endswith(gold[gi].lower()) or gold[gi].lower().endswith(rec[ri].lower()): indexes.append((gi,ri))
        if encodeds[0].first[i]>0: ri+=1
        if encodeds[0].second[i]>0: gi+=1
        if encodeds[0].first[i]==0 and encodeds[0].second[i]==0:
            ri+=1
            gi+=1
# le PB est que l'alignement peut "sauter" le premier (ou dernier) mot, donc les comptes peuvent etre differents
#    assert n1==len(rec)
#    assert n2==len(gold)
    return indexes


with open('trainbase.tab','w') as basef:
    with open('trainkey.tab','w') as keyf:
        for fich in trainfichs:
            fich="data/train/"+fich
            texte=""
            evs0=[]
            nev0=0
            with open(fich+".txt",'w') as g:
                print("=============================="+fich)
                with open(fich) as f: lines=f.readlines()
                txt = ""
                for l in lines: txt+=l.replace('«','"').strip()+' '
                # I dont know what SIGNAL is for, but it contains text
                txt=re.sub('<SIGNAL[^>]*>','',txt)
                txt=txt.replace('</SIGNAL>','')
                txt=re.sub('--+','',txt)
                txt=txt.replace('(','( ')
                txt=txt.replace(')',' )')
                txt=txt.strip()
                while True:
                    i=txt.find('<TEXT>')
                    if i<0: break
                    j=txt.find('</TEXT')
                    txt=txt[i+6:j]
                    while True:
                        i=txt.find('<')
                        if i<0: break
                        if i==0:
                            j=txt.find('>')
                            if txt[j-1]=='/': # il n'y a rien a afficher, ou evt vide
                                txt=txt[j+1:]
                            else: # chercher fin de l'elt
                                # 1er passage: pour treetagger, on n'ajoute pas l'info d'evt
                                if txt.lower().startswith("<event"):
                                    x=" ...EV "
                                    nev0+=1
                                else: x=""
                                txt=txt[j+1:]
                                i=txt.find('</')
                                if x.startswith(" ...EV"): evs0.append(txt[0:i])
                                x+=txt[0:i]+' '
                                g.write(txt[0:i]+' ')
                                texte+=x
                                j=txt.find('>')
                                txt=txt[j+1:]
                        else:
                            g.write(txt[:i])
                            texte+=txt[:i]
                            txt=txt[i:]
                    g.write(txt)
                    texte+=txt
            os.system(treetaggerpath+"/cmd/tree-tagger-french "+fich+".txt > "+fich+".tab")
            print(texte)

            texte=re.sub("  +"," ",texte)
            texte=re.sub("'","' ",texte)
            texte=texte.strip()
            txt=texte.split()
            toks0,tags0=[],[]
            nextisev=False
            for x in txt:
                if x=='...EV':
                    nextisev=True
                else:
                    toks0.append(x)
                    if nextisev: tags0.append('B')
                    else: tags0.append('O')
                    nextisev=False

            with open(fich+".tab") as f: lines=f.readlines()
            nev=0
            toks1,cols=[],[]
            with open(fich+".tab",'w') as f:
                for l in lines:
                    w=l.split()[0]
                    toks1.append(w)
                    cols.append(l.strip())

                idx=alignToks(toks0,toks1)
                # debug
#                print(toks0)
#                print(toks1)
#                print(idx)
#                for j0,j1 in idx: print(j0,toks0[j0],j1,toks1[j1])

                tags=['O' for i in range(len(toks1))]
                for i in range(len(idx)):
                    j1=idx[i][1]
                    j0=idx[i][0]
                    if tags0[j0]=='B':
                        tags[j1]='B'
                for i in range(len(toks1)):
                    f.write(cols[i]+'\t'+tags[i]+'\n')
                    basef.write(fich+'\t'+'0\t'+str(i)+'\t'+cols[i].split()[0]+'\n')
                    if tags[i]=='B':
                        nev+=1
                        keyf.write(fich+'\t'+'0\t'+str(i)+'\t'+'timex3\t'+'t'+str(nev)+'\t'+'1\t'+'type\t'+'EVENT\n')
            print("NEV",nev0,nev)
            evs1=[]
            for i in range(len(toks1)):
                if tags[i]=='B': evs1.append(toks1[i])
            for i in range(len(evs0)): print(evs0[i]+"\t"+evs1[i])
            assert nev0==nev

print("processing finished OK")

