import sys
import os
import re
from alignment.sequence import Sequence
from alignment.vocabulary import Vocabulary
from alignment.sequencealigner import SimpleScoring, GlobalSequenceAligner

treetaggerpath = "/home/xtof/softs/treetagger/"

reload(sys) # just to be sure
sys.setdefaultencoding('utf-8')

testfichs=("2003-01-15-0002.xml","2003-01-15-0005.xml","2003-01-15-0008.xml","algerie.xml","baldwin_frratrain_15.xml","bio_butler.xml","bove.xml","RDS020607SENATEURS.xml","texte5.xml")

def alignToks(gold,rec):
    r=Sequence(rec)
    g=Sequence(gold)
    v=Vocabulary()
    re=v.encodeSequence(r)
    rg=v.encodeSequence(g)
    scoring=SimpleScoring(2,-1)
    aligner=GlobalSequenceAligner(scoring, -2)
    score, encodeds = aligner.align(re, rg, backtrace=True)
    gi,ri=0,0
    err=0
    indexes=[]
    for i in range(len(encodeds[0])):
        if encodeds[0].first[i]>0 and encodeds[0].second[i]>0:
            # gi et ri sont alignes
            if rec[ri].lower() == gold[gi].lower(): indexes.append((gi,ri))
        if encodeds[0].first[i]>0: ri+=1
        if encodeds[0].second[i]>0: gi+=1
    return indexes


with open('testbase.tab','w') as basef:
    with open('testkey.tab','w') as keyf:
        with open('test0.tab','w') as recf:
            for fich in testfichs:
                texte=""
                nev0=0
                with open("data/test/"+fich+".txt",'w') as g:
                    print("=============================="+fich)
                    with open(fich) as f: lines=f.readlines()
                    txt = ""
                    for l in lines: txt+=l.strip()+' '
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
                                    # if x.startswith("...EV"): print("debugev "+txt[0:i])
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

                texte=re.sub("  +"," ",texte).strip()
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
                    print(toks0)
                    print(toks1)
                    print(idx)
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
                assert nev0==nev

                # stupid system that returns a segment for every verb
                assert len(toks1)==len(cols)
                nev=0
                for i in range(len(toks1)):
                    postag=cols[i].split()[1]
                    if postag.startswith("VER"):
                        nev+=1
                        recf.write(fich+'\t'+'0\t'+str(i)+'\t'+'timex3\t'+'t'+str(nev)+'\t'+'1\t'+'type\t'+'EVENT\n')

print("processing finished OK")

