import csv
import numpy as np
import operator
from collections import OrderedDict

def split_data(data, train_ratio=0.8):
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    n_train = int(np.floor(data.shape[0]*train_ratio))
    indices_train = indices[:n_train]
    indices_val = indices[n_train:]
    train = data[indices_train, :]
    validation = data[indices_val, :]
    return train, validation

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

def bayesianPredict(data_tst,dicPos, dicNeg, totalEnt, totalPos, totalNeg):
    result = np.array([])
    for eachData in data_tst[:,1]:
        separatedWords = eachData.split(' ')
        pos,neg=0,0
        for word in separatedWords:
            if dicNeg.has_key(word) or dicPos.has_key(word):
                try:
                    pos += np.log(dicPos[word]/totalPos)
                except:
                    pos += 0
                try:
                    neg += np.log(dicNeg[word]/totalNeg)
                except:
                    neg += 0
        pos += totalPos/totalEnt
        neg += totalNeg/totalEnt
       
        res = 1
        if pos<neg:
            res = 0
        result = np.insert(result,len(result),res)
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
    return efficiency


destFile='FinalStemmedSentimentAnalysisDataset.csv'
dictionaryPositives={}
dictionaryNegatives={}


#Used for counter
totalPositives=0
totalNegatives=0
totalEntries=0

#Metrics for compute algorithm efficiency

with open(destFile,'r') as destF:
    dataForIter = csv.reader(destF, 
                           delimiter = ';', 
                           quotechar = '"')

    data = [data for data in dataForIter]
data_array = np.asarray(data) 

print "Lectura Finalitzada"

ratio = 0.1
data_tr,data_tst = split_data(data_array,ratio)
print "particio finalitzada amb ratio de:",ratio
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
	else:
		for eachWord in eachEntry[1].split(' '):
			if (dictionaryPositives.has_key(eachWord)):
				dictionaryPositives[eachWord] += 1.0
        	else:
				dictionaryPositives[eachWord] = 1.0
		totalPositives+=1
	totalEntries+=1

print "Contar positius i negatius finalitzat"
#Sorts for the dicts
sortedListPositives =sorted(dictionaryPositives.items(), key=lambda x: x[1],reverse=True)
sortDictPositives = OrderedDict(sortedListPositives)
sortedListNegatives = sorted(dictionaryNegatives.items(), key=lambda x: x[1],reverse=True)
sortDictNegatives = OrderedDict(sortedListNegatives)

print "Ordenacio finalitzada"

#take some elements
numOfElements=100
nItems = OrderedDict(sortDictPositives.items()[:numOfElements])

print "Comencem a extreure resultats"
alpha = 0.2

resultBayes = bayesianPredict(data_tst,sortDictPositives,sortDictNegatives,totalEntries,totalPositives,totalNegatives)
resultLaplace = laplaceSmoothingPredict(data_tst,sortDictPositives,sortDictNegatives,totalEntries,totalPositives,totalNegatives,alpha)

print "resultats Finalitzats anem a evaluacio"

#get TP,TN,FP,FN
efficiencyBayes = evaluation(resultBayes,data_tst)
efficiencyLaplace = evaluation(resultLaplace,data_tst)

print "fi evaluacio"
print "_____________RESULTATS_______________"

#Some prints
print "Total positives: ", totalPositives
print "Total negatives: ", totalNegatives
print "Total entries", totalEntries
print "Array Resultat", result
print "efficiencyBayes",efficiencyBayes
print "efficiencyLaplace", efficiencyLaplace
#print "Dictionary Positives",sortDictPositives
print 
print '----------------------------------------'
print 
#print "Dictionary Negatives ",sortDictNegatives
print nItems['the']