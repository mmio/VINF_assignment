# https://github.com/explosion/spaCy/blob/01aec7a313753775603a9e6f752f75cc16ac43fb/examples/pipeline/multi_processing.py#L48

import os
import contextlib
from multiprocessing import Pool

from tqdm import tqdm
import json
from functools import partial
from collections import defaultdict

from spacy.util import minibatch

from files import urls, destination
from download import download_and_save
from prepare import get_text_from_gzip
from pipeline import get_pipe, get_tokenizer

from processors.histogramOfTokens import HistogramOfTokens
from processors.histogramOfQueries import HistogramOfQueries
from processors.dictionaryOfTokens import DictionaryOfTokens

def main():
    # open multiple files
    archives = [
        download_and_save(url, destination)
        for url in urls
    ]

    userIgnoreList = ['AnonID']

    openFilesList = dict()

    # 1. only divide into folders
    # for row in get_text_from_gzip(archives):
    #     userId = row[0]
    #     query = row[1]

    #     if userId in userIgnoreList:
    #         continue

    #     path = f'data/users/individual/{userId}'

    #     if not os.path.exists(path):
    #         os.makedirs(path)

    #     file = openFilesList.get(userId)

    #     if not file:
    #         openFilesList[userId] = open(f'{path}/queries.txt', 'a')
    #         file = openFilesList[userId]
    #     file.write(f'{query}\n')

    #     if len(openFilesList) == 1000:
    #         for f in openFilesList.values():
    #             f.close()
    #         openFilesList = dict()
            
    # 2. use the pipeline to tokenize and create processors to create stats
    processors = [
        # HistogramOfTokens(),
        # HistogramOfNormalized(),
        HistogramOfQueries('data/users/global/stats/hoq.json'),
        DictionaryOfTokens('data/users/global/stats/dict.pickle'),
        # DictionaryOfNormalized()
    ]
    d = DictionaryOfTokens('data/users/global/stats/dict.pickle')

    nlp = get_pipe()
    path = 'data/users/individual/'

    files = []
    outputs = []
    for folderName in tqdm(os.listdir(path)):
        break
        with open(f'{path}{folderName}/queries.txt', 'r') as file:

            # output = []
            
            for doc in nlp.pipe(file):

                # lst = [token.text for token in doc]
                # output.append(lst[:-1])

                d.add_doc(doc, folderName)

        # files.append(open(f'{path}{folderName}/tokenized.json', 'w'))
        # outputs.append(output)

        # if (len(files) == 1000):
        #     for f, o in zip(files, outputs):
        #         json.dump(o, f)
        #         f.close()
        #     files = []
        #     outputs = []

    if (len(files) > 0):
        for f, o in zip(files, outputs):
            json.dump(o, f)
        files = []
        outputs = []

    # for p in processors:
    #     p.save()
    # d.save()
    d.load()

    tokenizer = get_tokenizer(nlp)
    # 3. compare them to others / models
    for upperFolder in tqdm(os.listdir(path)):
        with open(f'{path}{upperFolder}/queries.txt') as file1:
            sim_dict = {}
            for row in tokenizer.pipe(file1):
            # for row in json.load(file1):
                for token in row:
                    if sim_dict.get(token.text, False):
                        print(f'Skipping {token.text}')
                        continue
                    sim_dict.update({token.text: True})

                    print(token.text)
                    print(len(list(d.get_item(token.text))))
                    for lowerFolder in tqdm(d.get_item(token.text)):
                        if lowerFolder == upperFolder:
                            continue
                        with open(f'{path}{lowerFolder}/queries.txt') as file2:
                            for doc in tokenizer.pipe(file2):
                                if token.text in doc.text:
                                    doc.similarity(row)
                                # this can be optimized
                                # pull out tokenizer, manually compare
                                # tokenizer(' '.join(row)).similarity(tokenizer(' '.join(row2)))

    # Collect global stats
    

    # nlp = get_pipe()
    # query_stream = map(lambda i: i[1], get_text_from_gzip(archives))
    # for doc in nlp.pipe(query_stream):
    #     for processor in processors:
    #         processor.add_doc(doc)

    # for p in processors:
    #     p.save()

    # d = defaultdict(list)
    # for row in get_text_from_gzip(archives):
    #     try:
    #         d[row[0]].append(row[1])
    #     except KeyError:
    #         d[row[0]] = [row[1]]
    # for k in d:
    #     if len(d[k]) > 300:
    #         print(k, len(d[k]))

    # fp = open('output/userQueries.json', 'w')
    # json.dump(d, fp)
    # fp.close()

    # d = defaultdict(list)
    # for row in get_text_from_gzip(archives):
    #     try:
    #         d[row[2]].append(row[1])
    #     except KeyError:
    #         d[row[2]] = [row[1]]

    # fp = open('output/timeQueries.json', 'w')
    # json.dump(d, fp)
    # fp.close()

    # d = defaultdict(list)
    # for row in get_text_from_gzip(archives):
    #     try:
    #         d[row[0]].append((row[1], row[2]))
    #     except KeyError:
    #         d[row[0]] = [(row[1], row[2])]

    # fp = open('output/userQueries.json', 'r')
    # json.dump(d, fp)
    # fp.close()

    # avg = []
    # counter = 0
    # with open('output/userQueries.json') as f:
    #     data = json.load(f)
    #     for item in data:
    #         print(len(data[item]))


        # data = json.load(f)
        # for doc in nlp.pipe(map(lambda r: r[0], data['479'])):
        #     if len(avg) == 0:
        #         avg = doc.vector
        #     else:
        #         avg += doc.vector
        #     counter += 1
        # print(avg/counter)

if __name__ == '__main__':
    main()







    # for upperFolder in tqdm(os.listdir(path)):
    #     with open(f'{path}{upperFolder}/tokenized.json') as file1:
    #         for row in tokenizer.pipe(map(lambda x: ' '.join(x), json.load(file1))):
    #         # for row in json.load(file1):
    #             # for token in row:
    #                 for lowerFolder in tqdm(d.get_item(token)):
    #                     if lowerFolder == upperFolder:
    #                         continue
    #                     with open(f'{path}{lowerFolder}/tokenized.json') as file2:
    #                         for doc in tokenizer.pipe(map(lambda x: ' '.join(x), json.load(file2))):
    #                             doc.similarity(row)
    #                             # this can be optimized
    #                             # pull out tokenizer, manually compare
    #                             # tokenizer(' '.join(row)).similarity(tokenizer(' '.join(row2)))
