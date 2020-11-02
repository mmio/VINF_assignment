# https://github.com/explosion/spaCy/blob/01aec7a313753775603a9e6f752f75cc16ac43fb/examples/pipeline/multi_processing.py#L48

import os
import numpy as np

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans

import pickle
import contextlib
from multiprocessing import Pool, Process
from functools import reduce
import matplotlib.pyplot as plt
import seaborn as sns

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


def chunks(l, n):
    """Yield n number of striped chunks from l."""
    for i in range(0, n):
        yield l[i::n]

def stuff0(folderNames, path, d, nlp):
    for folderName in folderNames:
        with open(f'{path}{folderName}/queries.txt', 'r') as file:
            for doc in nlp.pipe(file):
                d.add_doc(doc, folderName)

def stuff(upperFolders, path, d, tokenizer):
    for upperFolder in tqdm(upperFolders):
        with open(f'{path}{upperFolder}/queries.txt') as file1:
            sim_dict = {}
            for row in tokenizer.pipe(file1):
            # for row in json.load(file1):
                for token in row:
                    if sim_dict.get(token.text, False):
                        # print(f'Skipping {token.text}')
                        continue
                    sim_dict.update({token.text: True})

                    # print(token.text)
                    # print(len(list(d.get_item(token.text))))
                    for lowerFolder in d.get_item(token.text):
                        if lowerFolder == upperFolder:
                            continue
                        with open(f'{path}{lowerFolder}/queries.txt') as file2:
                            for doc in tokenizer.pipe(file2):
                                if token.text in doc.text:
                                    doc.similarity(row)
        print("folder finished")

def main():
    # open multiple files

    archives = [
        download_and_save(url)
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
    



    # get_all_query_docs_for_folder()
    # compute_average_vector_for_the_first_parts()
    # compute_average_for_end()

    # for_every_end -> go_through_the_folders_and find most similar based on similarity




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

    # jobs = []
    # for folderNames in chunks(os.listdir(path), 4):
    #     p = Process(target=stuff0, args=(folderNames, path, d, nlp,))
    #     jobs.append(p)
    #     p.start()

    # d.load()

    tokenizer = get_tokenizer(nlp)

    # exit(1)

    # jobs = []
    # for upperFolders in chunks(os.listdir(path), 4):
    #     p = Process(target=stuff, args=(upperFolders, path, d, tokenizer,))
    #     jobs.append(p)
    #     p.start()
    #3333. divide into train and test
    for folder in tqdm(os.listdir(path)):
        with open(f'{path}{folder}/queries.txt') as file:
            docs = []
            for doc in nlp.pipe(file, disable=['ner', 'tagger']):
                docs.append(doc.vector)
            # average and divide into 2 train.vector.pickle, test.vector.pickle
            percentil80 = int(len(docs) * 0.8)
            train = reduce(lambda a,b: a + b, docs[:percentil80], np.zeros(300)) / percentil80
            test = reduce(lambda a,b: a + b, docs[percentil80:], np.zeros(300)) / (len(docs) - percentil80)

            with open(f'{path}{folder}/vectors.pickle', 'wb') as vectors:
                pickle.dump(train, vectors, protocol=pickle.HIGHEST_PROTOCOL)
            with open(f'{path}{folder}/vectors.test.pickle', 'wb') as vectors_test:
                pickle.dump(test, vectors_test, protocol=pickle.HIGHEST_PROTOCOL)

    # 4444. tsne
    data_subset = []
    for folder in tqdm(os.listdir(path)):
        with open(f'{path}{folder}/vectors.pickle', 'rb') as vectors:
            try:
                x = pickle.load(vectors)
                array_sum = np.sum(x)
                if not np.isnan(array_sum):
                    data_subset.append(x)
            except:
                continue
    data_subset = data_subset[:1_000]
    print("start clustering")
    # clustering = DBSCAN(eps=0.5, min_samples=5, n_jobs=4).fit(data_subset)
    clustering = KMeans(n_clusters=20, random_state=0).fit(data_subset)
    print("finish clustering")

    print("pca start")
    pca_50 = PCA(n_components=50)
    pca_result_50 = pca_50.fit_transform(data_subset)
    print("pca finished")

    print("tsne start")
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000, n_jobs=4)
    tsne_results = tsne.fit_transform(pca_result_50)
    print("tsne finished")
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x=tsne_results[:,0], y=tsne_results[:,1], hue=clustering.labels_
    )
    plt.show()

    # 3. compare them to others / models
    # for upperFolder in tqdm(os.listdir(path)):
    #     with open(f'{path}{upperFolder}/queries.txt') as file1:
    #         sim_dict = {}
    #         for row in tokenizer.pipe(file1):
    #         # for row in json.load(file1):
    #             for token in row:
    #                 if sim_dict.get(token.text, False):
    #                     # print(f'Skipping {token.text}')
    #                     continue
    #                 sim_dict.update({token.text: True})

    #                 # print(token.text)
    #                 # print(len(list(d.get_item(token.text))))
    #                 for lowerFolder in tqdm(d.get_item(token.text)):
    #                     if lowerFolder == upperFolder:
    #                         continue
    #                     with open(f'{path}{lowerFolder}/queries.txt') as file2:
    #                         for doc in tokenizer.pipe(file2):
    #                             if token.text in doc.text:
    #                                 doc.similarity(row)






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
