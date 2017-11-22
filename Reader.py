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

destFile='shortDatabase.csv'
dictionaryPositives={}
dictionaryNegatives={}


#Used for counter
totalPositives=0
TotalNegatives=0
totalEntries=0

#Metrics for compute algorithm efficiency
efficiency={'TP':0,'FP':0,'TN':0,'FN':0}

with open(destFile,'r') as destF:
    dataForIter = csv.reader(destF, 
                           delimiter = ';', 
                           quotechar = '"')

    data = [data for data in dataForIter]
data_array = np.asarray(data) 

data_array = np.asarray(data) 

data_tr,data_tst = split_data(data_array)
print len(data_array)
print len(data_tr)
print len(data_tst)

#Some treaatment to create train and test sets

#Generation of the dicts
for eachEntry in data_tr:
	if eachEntry[-1]=='0':
		for eachWord in eachEntry[1].split(' '):
			if (dictionaryNegatives.has_key(eachWord)):
				dictionaryNegatives[eachWord] += 1.0
        	else:
				dictionaryNegatives[eachWord] = 1.0
		TotalNegatives+=1
	else:
		for eachWord in eachEntry[1].split(' '):
			if (dictionaryPositives.has_key(eachWord)):
				dictionaryPositives[eachWord] += 1.0
        	else:
				dictionaryPositives[eachWord] = 1.0
		totalPositives+=1
	totalEntries+=1

#Sorts for the dicts
sortedListPositives =sorted(dictionaryPositives.items(), key=lambda x: x[1],reverse=True)
sortDictPositives = OrderedDict(sortedListPositives)
sortedListNegatives = sorted(dictionaryNegatives.items(), key=lambda x: x[1],reverse=True)
sortDictNegatives = OrderedDict(sortedListNegatives)

#take some elements
numOfElements=100
nItems = OrderedDict(sortDictPositives.items()[:100])

#Some prints
print "Total positives: ", totalPositives
print "Total negatives: ", TotalNegatives
print "Total entries", totalEntries
print "Dictionary Positives",sortDictPositives
print 
print '----------------------------------------'
print 
print "Dictionary Negatives ",sortDictNegatives
print nItems['the']