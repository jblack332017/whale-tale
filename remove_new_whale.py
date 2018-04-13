import numpy as np
import pandas as pd
import sys
from shutil import copyfile
import csv


input = sys.argv[1]
output = sys.argv[2]

df = pd.read_csv(input + '/train.csv')

csv_file = open(output + '/train.csv', 'w')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Image', 'Id'])

for index, row in df.iterrows():
    if row['Id'] != "new_whale":
        copyfile(input + '/train/' + row['Image'], output+'/train/'+row['Image'])
        csv_writer.writerow([row['Image'], row['Id']])
