from os import listdir
from os.path import isfile, join
import sys
import shutil


INPUT_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]



train_df = pd.read_csv(INPUT_DIR + '/train.csv')

csv_file = open(OUTPUT_DIR + 'train.csv', 'w')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Image', 'Id'])

image_ids_pair = train_df['Id'].value_counts()
few_id_pairs = pd.Series(image_ids_pair).where(lambda x : x<5).dropna()
id_filename_pairs = {}
for id, value in few_id_pairs.items():
    id_filename_pairs[id] = list(train_df[train_df['Id'] == id]['Image'])
for id, file_names in id_filename_pairs.items():
    for i in range(5 - (len(file_names))):
        file_name =  random.choice(file_names)
        shutil.copy(INPUT_DIR + "/train/" + file_name, OUTPUT_DIR + "/train/" +str(i)+"_" + file_name)
        csv_writer.writerow([str(i)+"_" + file_name, id])
