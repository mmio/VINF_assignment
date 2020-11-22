import sys
import os
import numpy as np
from tqdm import tqdm
import datetime

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans, OPTICS, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.cluster import adjusted_rand_score

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

# def chunks(l, n):
#     """Yield n number of striped chunks from l."""
#     for i in range(0, n):
#         yield l[i::n]

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

def get_query(tsv_stream):
    for row in tsv_stream:
        yield row[1]

def queries_to_vector(nlp, tokenizer, tsv_stream):
    query_stream = map(lambda query: query[:-1], tsv_stream)
    
    vector_subset = []
    vector_norm_subset = []

    query_subset = []
    query_norm_subset = []
    
    last_query = ""
    for doc in tqdm(nlp.pipe(query_stream, disable=['ner', 'tagger'])):
        ## Keep only english queries
        if doc._.ld != 'en':
            continue

        ## Remove duplicate queries
        if doc.text == last_query:
            continue
        last_query = doc.text

        ## Try to fix spelling errors
        # for token in doc:
        #     if token._.hunspell_spell == False:
        #         print("found:", token.text)
        #         input()

        ## Remove oov and stopwords
        without_stopwords = [
            token.text
            for token in doc
            if not token.is_stop and not token.is_oov
        ]

        if len(without_stopwords) == 0:
            continue

        ## Normalized queries
        normalized = [
            token.lemma_
            for token in doc
            if not token.is_stop and not token.is_oov
        ]

        if len(normalized) == 0:
            continue

        q = ' '.join(without_stopwords)
        dq = tokenizer(q)
        vector_subset.append(dq.vector)
        query_subset.append(dq.text)

        q = ' '.join(normalized)
        dq = tokenizer(q)
        vector_norm_subset.append(dq.vector)
        query_norm_subset.append(dq.text)

    return vector_subset, vector_norm_subset, query_subset, query_norm_subset

def reduce_dimensions(data_subset, n_components):
    pca_of_n = PCA(n_components)
    return pca_of_n.fit_transform(data_subset)

def cluster_data(data, e, s):
    return AgglomerativeClustering(
        n_clusters=None,
        linkage='ward',
        compute_full_tree=True,
        distance_threshold=e,
        memory="/tmp/sklearn.tmp"
    ).fit(data)
    # return OPTICS(eps=e, min_samples=s, n_jobs=4).fit(data)
    # return DBSCAN(eps=0.01, min_samples=2, n_jobs=4).fit(data)
    # return KMeans(n_clusters=20).fit(data)

def save_scatterplot(savefile, x, y, hue):
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x=x, y=y, hue=hue
    )
    plt.savefig(savefile)

def vector_to_scatterplot(data_subset, query_subset, savefolder, sufix=''):

    # reduced_data = reduce_dimensions(data_subset, 50)
    reduced_data = data_subset
    labels = None
    # Grid search for parameters
    for e in [15]:
        for s in ['inf']:
            print(f'params e={e}, s={s}')

            clustered_data = cluster_data(reduced_data, e, s)

            os.mkdir(f'data/dates/{savefolder}/clusters{sufix}')
            for cluster_id, query in zip(clustered_data.labels_, query_subset):
                with open(f'data/dates/{savefolder}/clusters{sufix}/{cluster_id}', 'a') as f:
                    f.write(f'{query}\n')

            labels = clustered_data.labels_

            with open(f'data/dates/{savefolder}/e{e}-s{s}-result_queries{sufix}.txt', 'w') as f:
                for pair in zip(clustered_data.labels_, query_subset):
                    f.write(f'{str(pair)}\n')

            print("tsne start")
            tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000, n_jobs=8)
            tsne_results = tsne.fit_transform(reduced_data)
            print("tsne finished")

            with open(f'data/dates/{savefolder}/cluster_labels{sufix}.txt', 'w') as f:
                f.write(str(clustered_data.labels_))
            save_scatterplot(f'data/dates/{savefolder}/e{e}-s{s}-{savefolder}{sufix}.pdf', tsne_results[:,0], tsne_results[:,1], clustered_data.labels_)
    return labels

def skip_first_row(stream):
    next(stream)
    return stream

def time_to_datetime(str_time):
    return datetime.datetime.strptime(str_time, '%Y-%m-%d %H:%M:%S')

def divide_queries_based_on_time(tsv_stream):
    
    days = {}
    for row in tqdm(tsv_stream):

        if row[2] == 'QueryTime':
            continue

        time = time_to_datetime(row[2])

        if not days.get(f'{time.month} {time.day}', False):
            folder = f'data/dates/{time.month}_{time.day}'
            os.mkdir(folder)
            days.update({f'{time.month} {time.day}': open(f'{folder}/queries', 'a')})

        fp = days.get(f'{time.month} {time.day}')
        fp.write(f'{row[1]}\n')
    
    for value in days.values():
        value.close()

def main():
    archives = [
        download_and_save(url)
        for url in urls
    ]
    
    # userIgnoreList = ['AnonID']

    # textStatsCollectors = [
    #     HistogramOfQueries('data/users/global/stats/'),
    #     AverageQueryLength('data/users/global/stats/'),
    #     AverageNumberOfQueriesPerUser('data/users/global/stats/')
    # ]

    # docStatsCollectors = [
    #     HistogramOfTokens('data/users/global/stats/histogramOfTokens.json'),
    #     # DictionaryOfTokens('data/users/global/stats/dict.pickle'),
    # ]

    # path = 'data/users/individual/'

#    divide_queries_based_on_time(get_text_from_gzip(archives))

    nlp = get_pipe()
    tokenizer = get_tokenizer(nlp)

    path = f'data/dates/'
    for folder in tqdm(os.listdir(path)):
        if folder in sys.argv:
            data, data_norm, queries, queries_norm = queries_to_vector(nlp, tokenizer, open(f'{path}{folder}/queries', 'r'))
            labels = vector_to_scatterplot(data, queries, folder)
            labels_norm = vector_to_scatterplot(data_norm, queries_norm, folder, sufix='_norm')
            with open(f'{path}{folder}/cluster_similarity', 'w') as f:
                f.write(str(adjusted_rand_score(labels, labels_norm)))

    # queries_to_folders(get_text_from_gzip(archives), textStatsCollectors, userIgnoreList)
    # tokenize_queries(get_pipe(), path, docStatsCollectors)
    # cluster_user_history(path)
    # compare_users_cosim(path)

if __name__ == '__main__':
    main()
