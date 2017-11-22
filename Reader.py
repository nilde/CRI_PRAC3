import csv
import numpy as np
import operator
destFile='shortDatabase.csv'
dictionaryPositives={}
dictionaryNegatives={}

dictionaryPropNegatives={}
dictionaryPropPositives={}

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

#Some treaatment to create train and test sets

#Generation of the dicts
for eachEntry in data_array:
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
sortedDictionaryPositives = sorted(dictionaryPositives.items(), key=lambda x: x[1],reverse=True)
sortedDictionaryPositives=dict(sortedDictionaryPositives)

sortedDictionaryNegatives = sorted(dictionaryNegatives.items(), key=lambda x: x[1],reverse=True)
sortedDictionaryNegatives=dict(sortedDictionaryNegatives)



#Some prints
print "Total positives: ", totalPositives
print "Total negatives: ", TotalNegatives
print "Total entries", totalEntries
print "Dictionary Positives",sortedDictionaryPositives
print 
print 
print "Dictionary Negatives ",sortedDictionaryNegatives