import csv
import numpy as np
import operator
from collections import OrderedDict
import time
from sklearn.cross_validation import StratifiedShuffleSplit

def split_data(data, train_ratio=0.8):
    stratSplit = StratifiedShuffleSplit(data[1:,-1],1, train_size=train_ratio,random_state=42)
    for train_idx,test_idx in stratSplit:
        train = data[train_idx]
        validation = data[test_idx]

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
    print "tp: "+str(tp)+" tn: "+str(tn)+" fp: "+str(fp)+" fn: "+str(fn)
    print "Precision: "+str(prec)+"\nAccuracy: "+str(acc)+"\nRecall: "+str(rec)+"\nSpecificity: "+str(spcy)+"\nfScore: "+str(fS)
    return prec, acc, rec, spcy, fS

def laplaceSmoothingPredict(data_tst,dicPos, dicNeg, totalEnt, totalPos, totalNeg):
    result = np.array([])
    for eachData in data_tst[:,1]:
        separatedWords = eachData.split(' ')
        pos,neg=0,0
        for word in separatedWords:
            if dicNeg.has_key(word) or dicPos.has_key(word):
                try:
                    pos += dicPos[word]
                except:
                    pos += 0
                try:
                    neg += dicNeg[word]
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
                pos += dicPos[word]
            else:
                pos += 0
            if dicNeg.has_key(word):
                neg += dicNeg[word]
            else:
                neg += 0

        pos += (totalPosExamples/totalEnt)*(-1)
        neg += (totalNegExamples/totalEnt)*(-1)

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

def main():
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
    #print data
    data_array = np.asarray(data)
    total_Pos = 0
    totalData = 0
    for data in data_array:
        if data[3] == '1':
            total_Pos += 1
        totalData += 1
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
            totalNegativeEntries+=1
        else:
            for eachWord in eachEntry[1].split(' '):
                if (dictionaryPositives.has_key(eachWord)):
                    dictionaryPositives[eachWord] += 1.0
                else:
                    dictionaryPositives[eachWord] = 1.0
                totalPositives+=1
            totalPositiveEntries+=1
        totalEntries+=1
    
    #Bayes
    """for eachWord in dictionaryPositives.keys():
        dictionaryPositives[eachWord]=np.log(dictionaryPositives[eachWord]/totalPositives)
    for eachWord in dictionaryNegatives.keys():
        dictionaryNegatives[eachWord]=np.log(dictionaryNegatives[eachWord]/totalNegatives)
    """ 
    
    #laplace
    alpha = 0.5
    dictionaryPositivesLS = {}
    dictionaryNegativesLS = {}
    for eachWord in dictionaryPositives.keys():
        dictionaryPositivesLS[eachWord]= np.log((dictionaryPositives[eachWord]+alpha)/(totalPositives+alpha*totalEntries))
    for eachWord in dictionaryNegatives.keys():
        dictionaryNegativesLS[eachWord]= np.log((dictionaryNegatives[eachWord]+alpha)/(totalNegatives+alpha*totalEntries))
    
    t1 = time.time()
    print "Analisi train en",t1-t0
    print "Contar positius i negatius finalitzat (45%)"
    #Sorts for the dicts
    #bayes
    """sortedListPositives =sorted(dictionaryPositives.items(), key=lambda x: x[1],reverse=True)
    sortDictPositives = OrderedDict(sortedListPositives)
    sortedListNegatives = sorted(dictionaryNegatives.items(), key=lambda x: x[1],reverse=True)
    sortDictNegatives = OrderedDict(sortedListNegatives)
    """
    #laplace
    sortedListPositivesLS =sorted(dictionaryPositivesLS.items(), key=lambda x: x[1],reverse=True)
    sortDictPositivesLS = OrderedDict(sortedListPositivesLS)
    sortedListNegativesLS = sorted(dictionaryNegativesLS.items(), key=lambda x: x[1],reverse=True)
    sortDictNegativesLS = OrderedDict(sortedListNegativesLS)
    
    t2 = time.time()
    print "Ordenacio finalitzada (60%) temps",t2-t1
    
    #take some elements
    numOfElements=100
    #firstPositives = OrderedDict(sortDictPositives.items()[:numOfElements])
    #firstNegatives = OrderedDict(sortDictNegatives.items()[:numOfElements])
    firstPositivesLS = OrderedDict(sortDictPositivesLS.items()[:numOfElements])
    firstNegativesLS = OrderedDict(sortDictNegativesLS.items()[:numOfElements])
    t3=time.time()
    print "Comencem a extreure resultats (75%)"
    
    #resultBayes = bayesianPredict(data_tst,sortDictPositives,sortDictNegatives,totalEntries,totalPositives,totalNegatives,totalPositiveEntries,totalNegativeEntries)
    resultLaplace = laplaceSmoothingPredict(data_tst,sortDictPositivesLS,sortDictNegativesLS,totalEntries,totalPositives,totalNegatives)
    
    #fixedDictionarySize
    """totalPositivesFixed = abs(sum(firstPositives.values()))
    totalNegativesFixed = abs(sum(firstNegatives.values()))
    resultBayes = bayesianPredict(data_tst,firstPositives,firstNegatives,numOfElements,totalPositivesFixed,totalNegativesFixed,totalPositiveEntries,totalNegativeEntries)
    totalPositivesFixedLS = abs(sum(firstPositivesLS.values()))
    totalNegativesFixedLS = abs(sum(firstNegativesLS.values()))
    resultBayes = laplaceSmoothingPredict(data_tst,firstPositivesLS,firstNegativesLS,numOfElements,totalPositivesFixedLS,totalNegativesFixedLS,totalPositiveEntries,totalNegativeEntries)
    
    """
    t4 = time.time()
    print "resultats Finalitzats anem a evaluacio (90%) amb temps",t4-t3
    
    #get TP,TN,FP,FN
    #efficiencyBayes = evaluation(resultBayes,data_tst)
    efficiencyLaplace = evaluation(resultLaplace,data_tst)
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
    #print "Array resultBayes", resultBayes
    print "Array resultLaplace", resultLaplace
    #print "Resultats Bayes"
    #printMedidas(efficiencyBayes['TP'],efficiencyBayes['TN'],efficiencyBayes['FP'],efficiencyBayes['FN'])
    print "Resultats Laplace"
    printMedidas(efficiencyLaplace['TP'],efficiencyLaplace['TN'],efficiencyLaplace['FP'],efficiencyLaplace['FN'])
    print 
    print '----------------------------------------'
    print 
    #print "Dictionary Negatives ",sortDictNegatives

if __name__ == "__main__":
    
    main()
    
    
    