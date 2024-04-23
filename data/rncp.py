import wget
import pandas as pd
import numpy as np
import os
from clean_text import clean_text
import string
from sklearn.model_selection import train_test_split

# This file takes RNCP dataset (from rncp.csv file) and puts the different classes in the right folders
def dump_data(phase, name, text, path='./rncp_od/'):
    if phase == "train":
        full_path = os.path.join(path, phase)+"/0"
    else:
        full_path = os.path.join(path, phase)

    if not os.path.exists(full_path):
        os.makedirs(full_path)

    with open(f'{full_path}/{name}.txt', 'w') as fp:
        fp.write(text)
        print('Succesfully written', f'{full_path}/{name}.txt')

def export_ds(subsets, phase):
    raw_text = {
        "1-environnement" : '',
        "2-defense" : '',
        "3-patrimoine": '',
        "4-economie": '',
        "5-recherche": '',
        "6-nautisme": '',
        "7-aeronautique": '',
        "8-securite": '',
        "9-multimedia": '',
        "10-humanitaire": '',
        "11-nucleaire": '',
        "12-enfance": '',
        "13-saisonnier": '',
        "14-assistance": '',
        "15-sport": '',
        "16-ingenierie": '',
    }

    for data_label, subset in zip(subsets, raw_text):
        for text in subsets[data_label]:
            text = clean_text(text,language = 'french')
            raw_text[subset] += f'\n\n{text}'

    for subset in raw_text:
        dump_data(phase, subset, raw_text[subset])

rncp = pd.read_csv('./rncp.csv')
Y = rncp[['Categorie']]
X = rncp[['text_certifications']]
xTrain, xTest, yTrain, yTest = train_test_split(X, Y,
                                                    test_size=0.25,
                                                    random_state=0,
                                                    stratify=Y)

rncpTrain = pd.concat([xTrain,yTrain],axis = 1)
rncpTest = pd.concat([xTest,yTest],axis = 1)

rncpTrain = rncpTrain.rename(columns={"text_certifications": "description", "Categorie": "label"})
rncpTest = rncpTest.rename(columns={"text_certifications": "description", "Categorie": "label"})

subsets_test = {
    "1": [],
    "2": [],
    "3": [],
    "4": [],
    "5": [],
    "6": [],
    "7": [],
    "8": [],
    "9": [],
    "10": [],
    "11": [],
    "12": [],
    "13": [],
    "14": [],
    "15": [],
    "16": [],
}

for idx, el in enumerate(np.array(rncpTest)):
    label = el[1]
    text = el[0]
    subsets_test[f'{label}'].append(text)

print('Total samples (test)', idx+1)

export_ds(subsets_test, 'test')

subsets_train = {
    "1": [],
    "2": [],
    "3": [],
    "4": [],
    "5": [],
    "6": [],
    "7": [],
    "8": [],
    "9": [],
    "10": [],
    "11": [],
    "12": [],
    "13": [],
    "14": [],
    "15": [],
    "16": [],
}

for idx, el in enumerate(np.array(rncpTrain)):
    label = el[1]
    text = el[0]
    subsets_train[f'{label}'].append(text)

print("Total samples (train):", idx+1)

export_ds(subsets_train, 'train')