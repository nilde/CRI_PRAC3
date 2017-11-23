import csv
import numpy as np
import operator
from collections import OrderedDict
import time

def split_data(data, train_ratio=0.8):
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    n_train = int(np.floor(data.shape[0]*train_ratio))
    indices_train = indices[:n_train]
    indices_val = indices[n_train:]
    train = data[indices_train, :]
    validation = data[indices_val, :]
    return train, validation

def precision(tPositive, fPositive):
    return tPositive/(tPositive+fPositive)

def accuracy(tPositive, tNegative, fPositive, fNegative):
    return (tPositive+tNegative)/(tPositive+tNegative+fPositive+fNegative)

def recall(tPositive, fNegative):
    if (tPositive+fNegative) == 0:
        return 0
    else:
        return tPositive/(tPositive+fNegative)

def specifity(tNegative, fPositive):
    if (tNegative+fPositive) == 0:
        return 0
    else:
        return tNegative/(tNegative+fPositive)

def fScore(prec, rec):
    if (prec+rec) == 0:
        return 0
    else:
        return (2*prec*rec) /(prec+rec)

def printMedidas(tp, tn, fp, fn):
    tp = float(tp)
    tn = float(tn)
    fp = float(fp)
    fn = float(fn)
    
    prec = precision(tp,fp)
    acc = accuracy(tp,tn,fp,fn)
    rec = recall(tp,fn)
    spcy = specifity(tn,fp)
    fS = fScore(prec, rec)
    print  "tp: "+str(tp)+" tn: "+str(tn)+" fp: "+str(fp)+" fn: "+str(fn)
    print "Precision: "+str(prec)+"\nAccuracy: "+str(acc)+"\nRecall: "+str(rec)+"\nSpecificity: "+str(spcy)+"\nfScore: "+str(fS)
    return prec, acc, rec, spcy, fS

def laplaceSmoothingPredict(data_tst,dicPos, dicNeg, totalEnt, totalPos, totalNeg, alpha=0.2):
    result = np.array([])
    for eachData in data_tst[:,1]:
        separatedWords = eachData.split(' ')
        pos,neg=0,0
        for word in separatedWords:
            if dicNeg.has_key(word) or dicPos.has_key(word):
                try:
                    pos += np.log((dicPos[word]+alpha)/(totalPos+alpha*totalEnt))
                except:
                    pos += 0
                try:
                    neg += np.log((dicNeg[word]+alpha)/(totalNeg+alpha*totalEnt))
                except:
                    neg += 0
        pos += totalPos/totalEnt
        neg += totalNeg/totalEnt
       
        res = 1
        if pos<neg:
            res = 0
        result = np.insert(result,len(result),res)
    return result

def bayesianPredict(data_tst,dicPos, dicNeg, totalEnt, totalPos, totalNeg,totalPosExamples,totalNegExamples):
    result = np.array([])
    positives=0
    negatives=0
    for eachData in data_tst[:,1]:
        separatedWords = eachData.split(' ')
        pos,neg=0,0
        for word in separatedWords:
            if dicPos.has_key(word):
                pos += np.log(dicPos[word]/totalPos)
            else:
                pos += 0
            if dicNeg.has_key(word):
                neg += np.log(dicNeg[word]/totalNeg)
            else:
                neg += 0

        pos += totalPosExamples/totalEnt
        neg += totalNegExamples/totalEnt

        res = 1
        if pos<neg:
            res = 0

        if res==1:
            positives+=1
        else:
            negatives+=1

        result = np.insert(result,len(result),res)
    print 'Number of positives classifications: ',positives
    print 'Number of negatives classifications: ',negatives
    return result

def evaluation(predict, validatation):
    efficiency={'TP':0,'FP':0,'TN':0,'FN':0}
    for predicted,value in zip(predict,validatation[:,-1]):
        if str(int(predicted)) == value:
            if predicted == 1.0:
                efficiency['TP']+=1
            else:
                efficiency['TN']+=1
        else:
            if predicted == 1.0:
                efficiency['FP']+=1
            else:
                efficiency['FN']+=1
    return dict(efficiency)


destFile='FinalStemmedSentimentAnalysisDataset.csv'
#destFile='shortDatabase.csv'
dictionaryPositives={}
dictionaryNegatives={}


#Used for word count
totalPositives=0
totalNegatives=0

#Used for total count
totalPositiveEntries=0
totalNegativeEntries=0
totalEntries=0

#Metrics for compute algorithm efficiency

with open(destFile,'r') as destF:
    dataForIter = csv.reader(destF, 
                           delimiter = ';', 
                           quotechar = '"')

    data = [data for data in dataForIter]
data_array = np.asarray(data)

print "Lectura Finalitzada (15%)"

ratio = 0.8
data_tr,data_tst = split_data(data_array,ratio)
print "Particio finalitzada amb ratio de:",ratio,' (30%)'
t0 = time.time()
#Some treaatment to create train and test sets

#Generation of the dicts
for eachEntry in data_tr:
    if eachEntry[-1]=='0':
        for eachWord in eachEntry[1].split(' '):
            if (dictionaryNegatives.has_key(eachWord)):
                dictionaryNegatives[eachWord] += 1.0
            else:
                dictionaryNegatives[eachWord] = 1.0
            totalNegatives+=1
        totalPositiveEntries+=1
    else:
        for eachWord in eachEntry[1].split(' '):
            if (dictionaryPositives.has_key(eachWord)):
                dictionaryPositives[eachWord] += 1.0
            else:
                dictionaryPositives[eachWord] = 1.0
            totalPositives+=1
        totalNegativeEntries+=1
    totalEntries+=1
t1 = time.time()
print "Analisi train en",t1-t0
print "Contar positius i negatius finalitzat (45%)"
#Sorts for the dicts
sortedListPositives =sorted(dictionaryPositives.items(), key=lambda x: x[1],reverse=True)
sortDictPositives = OrderedDict(sortedListPositives)
sortedListNegatives = sorted(dictionaryNegatives.items(), key=lambda x: x[1],reverse=True)
sortDictNegatives = OrderedDict(sortedListNegatives)

t2 = time.time()
print "Ordenacio finalitzada (60%) temps",t2-t1

#take some elements
numOfElements=100
nItems = OrderedDict(sortDictPositives.items()[:numOfElements])
t3=time.time()
print "Comencem a extreure resultats (75%)"
alpha = 0.5

resultBayes = bayesianPredict(data_tst,sortDictPositives,sortDictNegatives,totalEntries,totalPositives,totalNegatives,totalPositiveEntries,totalNegativeEntries)
#resultLaplace = laplaceSmoothingPredict(data_tst,sortDictPositives,sortDictNegatives,totalEntries,totalPositives,totalNegatives,alpha)

t4 = time.time()
print "resultats Finalitzats anem a evaluacio (90%) amb temps",t4-t3

#get TP,TN,FP,FN
efficiencyBayes = evaluation(resultBayes,data_tst)
#efficiencyLaplace = evaluation(resultLaplace,data_tst)
t5 = time.time()


print "fi evaluacio (100%) va passar",t5-t0
print
print "_____________RESULTATS_______________"
print 
#Some prints
print "Total positives words: ", totalPositives
print "Total negatives words: ", totalNegatives
print "Total positives examples: ", totalPositiveEntries
print "Total negatives examples: ", totalNegativeEntries
print "Total entries", totalEntries
print "Array resultBayes", resultBayes
#print "Array resultLaplace", resultLaplace
print "Resultats Bayes"
printMedidas(efficiencyBayes['TP'],efficiencyBayes['TN'],efficiencyBayes['FP'],efficiencyBayes['FN'])
#print "Resultats Laplace"
#printMedidas(efficiencyLaplace['TP'],efficiencyLaplace['TN'],efficiencyLaplace['FP'],efficiencyLaplace['FN'])
print 
print '----------------------------------------'
print 
#print "Dictionary Negatives ",sortDictNegatives