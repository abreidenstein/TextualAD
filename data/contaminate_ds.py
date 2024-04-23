import os
from os import listdir
from os.path import isfile, join
import random
from pathlib import Path


# This file creates contaminated training dataset for unsupervised setting
def contaminate_ds(root, clean_file, contamination, write=True):
    all_files = [f for f in listdir(root+"/0") if isfile(join(root+"/0", f))]
    all_files.remove(clean_file)

    clean_fp = open(join(root+"/0", clean_file), 'r')
    clean_file_lines = clean_fp.read().split('\n')
    clean_file_lines = [line for line in clean_file_lines if line not in ['', ' ', '\n']]

    anomalies = []
    for anom_name in all_files:
        anom_fp = open(join(root+"/0", anom_name), 'r')
        anomalies.extend(anom_fp.read().split('\n'))

    anomalies = [line for line in anomalies if line not in ['', ' ', '\n']]

    random.shuffle(anomalies)

    no_entries_anomaly = int( (100 * len(clean_file_lines))/(100-contamination) ) - len(clean_file_lines)

    clean_file_lines.extend(anomalies[:no_entries_anomaly])
    random.shuffle(clean_file_lines)

    if write:
        contaminated_ds = ''
        for item in clean_file_lines:
            if item!='\n':
                contaminated_ds += f'{item}\n'
            else:
                print('newline ignored')

        if not os.path.exists(root + '/' + str(contamination)):
            os.makedirs(root + '/' + str(contamination))
        Path(root + '/' + str(contamination) + '/' + clean_file[:-4] + '.txt').touch()
        with open(root + '/' + str(contamination) + '/' + clean_file[:-4] + '.txt', 'w') as f:
            f.write(contaminated_ds)

    

     
random.seed(0)
# 20Newsgroups
phases = ['train']
ag_subsets = ['comp', 'misc', 'pol', 'rec', 'rel', 'sci']
contamination = [5, 10, 15, 20, 25]

for phase in phases:
    for subset in ag_subsets:
        print('Contaminating : 20 Newsgroups -', subset)
        for cont in contamination:
            contaminate_ds(f'./20ng_od/{phase}/', f'{subset}.txt', cont)
print('\n')

# Rncp
phases = ['train']
rncp_subsets = ['1-environnement','2-defense','3-patrimoine','4-economie',
                        '5-recherche','6-nautisme','7-aeronautique','8-securite',
                        '9-multimedia','10-humanitaire','11-nucleaire','12-enfance',
                        '13-saisonnier','14-assistance','15-sport','16-ingenierie']
contamination = [5, 10, 15, 20, 25]

for phase in phases:
    for subset in rncp_subsets:
        print('Contaminating : RNCP -', subset)
        for cont in contamination:
            contaminate_ds(f'./rncp_od/{phase}', f'{subset}.txt', cont)
print('\n')

# AG News
phases = ['train']
ag_subsets = ['business', 'sci', 'sports', 'world']
contamination = [5, 10, 15, 20, 25]

for phase in phases:
    for subset in ag_subsets:
        print('Contaminating : AG News -', subset)
        for cont in contamination:
            contaminate_ds(f'./ag_od/{phase}', f'{subset}.txt', cont)
