import csv
import numpy as np
with open(dest_file,'r') as dest_f:
    data_iter = csv.reader(dest_f, 
                           delimiter = delimiter, 
                           quotechar = '"')
    data = [data for data in data_iter]
data_array = np.asarray(data)    