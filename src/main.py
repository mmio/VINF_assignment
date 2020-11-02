import os
import numpy as np
from tqdm import tqdm

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.pairwise import cosine_similarity

import pickle
import contextlib
from multiprocessing import Pool, Process
from functools import reduce
import matplotlib.pyplot as plt
import seaborn as sns

from files import urls, destination
from download import download_and_save
from prepare import get_text_from_gzip
from pipeline import get_pipe, get_tokenizer

from processors.histogramOfTokens import HistogramOfTokens
from processors.histogramOfQueries import HistogramOfQueries
from processors.dictionaryOfTokens import DictionaryOfTokens
from processors.averageQueryLength import AverageQueryLength
from processors.averageNumberOfQueriesPerUser import AverageNumberOfQueriesPerUser 


def chunks(l, n):
    """Yield n number of striped chunks from l."""
    for i in range(0, n):
        yield l[i::n]

def queries_to_folders(tsv_stream, textStatsCollectors, userIgnoreList):
    openFilesList = dict()
    for row in tqdm(tsv_stream):
        userId = row[0]
        query = row[1]

        if userId in userIgnoreList:
            continue

        for tsc in textStatsCollectors:
            tsc.add_doc(query, userId)

        path = f'data/users/individual/{userId}'

        if not os.path.exists(path):
            os.makedirs(path)

        file = openFilesList.get(userId)

        if not file:
            openFilesList[userId] = open(f'{path}/queries.txt', 'a')
            file = openFilesList[userId]
        file.write(f'{query}\n')

        if len(openFilesList) == 1000:
            for f in openFilesList.values():
                f.close()
            openFilesList = dict()

    for tsc in textStatsCollectors:
        tsc.save()

def tokenize_queries(nlp, path, docStatsCollectors):
    files = []
    outputs = []
    for folderName in tqdm(os.listdir(path)):
        with open(f'{path}{folderName}/queries.txt', 'r') as file:

            output = []
            docs = []
            
            for doc in nlp.pipe(file, disable=['ner', 'tagger']):

                lst = [token.text for token in doc]
                output.append(lst[:-1])
                docs.append(doc.vector)

                for dsc in docStatsCollectors:
                    dsc.add_doc(doc, folderName)

            percentil80 = int(len(docs) * 0.8)
            train = reduce(lambda a,b: a + b, docs[:percentil80], np.zeros(300)) / percentil80
            test = reduce(lambda a,b: a + b, docs[percentil80:], np.zeros(300)) / (len(docs) - percentil80)

            with open(f'{path}{folderName}/vectors.pickle', 'wb') as vectors:
                pickle.dump(train, vectors, protocol=pickle.HIGHEST_PROTOCOL)
            with open(f'{path}{folderName}/vectors.test.pickle', 'wb') as vectors_test:
                pickle.dump(test, vectors_test, protocol=pickle.HIGHEST_PROTOCOL)

        files.append(open(f'{path}{folderName}/tokenized.pickle', 'wb'))
        outputs.append(output)

        if (len(files) == 1000):
            for f, o in zip(files, outputs):
                pickle.dump(o, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.close()
            files = []
            outputs = []

    if (len(files) > 0):
        for f, o in zip(files, outputs):
            pickle.dump(o, f, protocol=pickle.HIGHEST_PROTOCOL)
        files = []
        outputs = []

    for dsc in docStatsCollectors:
        dsc.save()

def cluster_user_history(path):
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

    print("pca start")
    pca_50 = PCA(n_components=50)
    pca_result_50 = pca_50.fit_transform(data_subset)
    print("pca finished")

    print("start clustering")
    clustering = DBSCAN(eps=0.1, min_samples=5, n_jobs=4).fit(pca_result_50)
    # clustering = KMeans(n_clusters=100).fit(data_subset)
    print("finish clustering")


    print("tsne start")
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000, n_jobs=4)
    tsne_results = tsne.fit_transform(pca_result_50)
    print("tsne finished")
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x=tsne_results[:,0], y=tsne_results[:,1], hue=clustering.labels_
    )
    plt.savefig('clustering.pdf')

def compare_users_cosim(path):
    others = []
    for folder in tqdm(os.listdir(path)):
        other = open(f'{path}{folder}/vectors.pickle', 'rb')
        y = pickle.load(other)
        if np.isnan(y[0]):
            continue
        others.append((y, folder))
    
    xs = []
    for folder in tqdm(os.listdir(path)):
        test = open(f'{path}{folder}/vectors.test.pickle', 'rb')

        x = pickle.load(test)
        if np.isnan(x[0]):
            continue
        xs.append((x, folder))

    def pr(xs, others, path):
        for vec, name in tqdm(xs):
            with open(f'{path}{name}/similar.txt', 'a') as s1:
                y, names = zip(*others)
                sims = cosine_similarity([vec], y)
                for sim, folder in zip(sims, names):
                    # s1.write(f'{sim} {folder}\n')
                    # with open(f'{path}{folder}/similar.txt', 'a') as s2:
                    #   s2.write(f'{sim} {name}\n')
                    ...
                        
    jobs = []
    for i, chunk in enumerate(chunks(xs, 4)):
        p = Process(target=pr, args=(chunk, others[i * len(chunk):],path,))
        jobs.append(p)
        p.start()

def main():
    archives = [
        download_and_save(url)
        for url in urls
    ]
    
    userIgnoreList = ['AnonID']

    textStatsCollectors = [
        HistogramOfQueries('data/users/global/stats/'),
        AverageQueryLength('data/users/global/stats/'),
        AverageNumberOfQueriesPerUser('data/users/global/stats/')
    ]

    docStatsCollectors = [
        HistogramOfTokens('data/users/global/stats/histogramOfTokens.json'),
        # DictionaryOfTokens('data/users/global/stats/dict.pickle'),
    ]

    path = 'data/users/individual/'

    # queries_to_folders(get_text_from_gzip(archives), textStatsCollectors, userIgnoreList)
    # tokenize_queries(get_pipe(), path, docStatsCollectors)
    # cluster_user_history(path)
    compare_users_cosim(path)

if __name__ == '__main__':
    main()