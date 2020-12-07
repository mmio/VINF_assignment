import sys
import os
import numpy as np
from tqdm import tqdm
import datetime

import concurrent.futures

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans, OPTICS, AgglomerativeClustering, Birch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.feature_extraction.text import TfidfVectorizer

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

from MultipleOpenFiles import MultipleOpenFiles
from joblib import Memory

def get_query(tsv_stream):
    for row in tsv_stream:
        yield row[1]

def queries_to_vector(nlp, tokenizer, tsv_stream):
    # hot = HistogramOfQueries('data/global_stats/')
    query_stream = map(lambda query: query[:-1], tsv_stream)
    
    vector_subset = []
    vector_norm_subset = []

    query_subset = []
    query_norm_subset = []

    uniq_queries = set(query_stream)

    for doc in tqdm(nlp.pipe(uniq_queries, disable=['ner', 'tagger'])):
        ## Keep only english queries
        if doc._.ld != 'en':
            continue

        ## Try to fix spelling errors
        # for token in doc:
        #     if token._.hunspell_spell == False:
        #         print("found:", token.text)
        #         input()

        ## Remove oov and stopwords
        # without_stopwords = [
        #     token.text
        #     for token in doc
        #     if not token.is_stop and not token.is_oov
        # ]

        # if len(without_stopwords) == 0:
        #     continue

        # ## Normalized queries
        # normalized = [
        #     token.lemma_
        #     for token in doc
        #     if not token.is_stop and not token.is_oov
        # ]

        normalized = []
        without_stopwords = []
        for token in doc:
            if not token.is_stop and not token.is_oov:
                normalized.append(token.lemma_)
                without_stopwords.append(token.text)

        if len(normalized) == 0:
            continue

        ## collect stats for the day
        # hot.add_doc(doc, 0)

        q = ' '.join(without_stopwords)
        dq = tokenizer(q)
        vector_subset.append(dq.vector)
        query_subset.append(dq.text)

        q = ' '.join(normalized)
        dq = tokenizer(q)
        vector_norm_subset.append(dq.vector)
        query_norm_subset.append(dq.text)

    # hot.save()
    # hot = None
    return vector_subset, vector_norm_subset, query_subset, query_norm_subset

def reduce_dimensions(data_subset, n_components):
    pca_of_n = PCA(n_components)
    return pca_of_n.fit_transform(data_subset)

def cluster_data(data, e, s):
    # return AgglomerativeClustering(
    #     n_clusters=None,
    #     linkage='ward',
    #     compute_full_tree=True,
    #     distance_threshold=e,
    #     memory=Memory('./cachedir', verbose=0)
    # ).fit(data)
    return Birch(n_clusters=500).fit(data)
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
    labels = []
    # avg_similarity_of_clusters = []
    # Grid search for parameters
    for e in [5]:
        for s in ['inf']:
            # print(f'params e={e}, s={s}')
            if len(reduced_data) == 0:
                break

            clustered_data = cluster_data(reduced_data, e, s)

            os.mkdir(f'data/dates/{savefolder}/clusters{sufix}')
            for cluster_id, query in zip(clustered_data.labels_, query_subset):
                with open(f'data/dates/{savefolder}/clusters{sufix}/{cluster_id}', 'a') as f:
                    f.write(f'{query}\n')

            labels = clustered_data.labels_

            with open(f'data/dates/{savefolder}/e{e}-s{s}-result_queries{sufix}.txt', 'w') as f:
                for pair in zip(clustered_data.labels_, query_subset):
                    f.write(f'{str(pair)}\n')

            # print("tsne start")
            # tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000, n_jobs=1)
            # tsne_results = tsne.fit_transform(reduced_data)
            # print("tsne finished")

            # with open(f'data/dates/{savefolder}/cluster_labels{sufix}.txt', 'w') as f:
            #     f.write(str(clustered_data.labels_))
            # save_scatterplot(f'data/dates/{savefolder}/e{e}-s{s}-{savefolder}{sufix}.pdf', tsne_results[:,0], tsne_results[:,1], clustered_data.labels_)
    
            # save cluster data to folder, for further comparison
            # avgs = []
            os.mkdir(f'data/dates/{savefolder}/cluster_dump{sufix}')

            for label in set(labels):
                with open(f'data/dates/{savefolder}/cluster_dump{sufix}/{label}', 'wb') as fh:
                    for i in range(len(labels)):
                        if labels[i] == label:
                            # vecs.append(data_subset[i])
                            pickle.dump(data_subset[i], fh)
                # print(label)

                # save/dump(label, vecs)
                # a potom porovnam iba token
                # with open(f'data/dates/{savefolder}/cluster_dump{sufix}/{label}', 'wb') as fh:
                #     pickle.dump(vecs, fh)
                
            #     sims = cosine_similarity(vecs)

            #     avg = np.mean(list(map(lambda x: np.mean(x), sims)))
            #     avgs.append(avg)
            # print(np.mean(avgs))

    return labels

# def skip_first_row(stream):
#     next(stream)
#     return stream

def isotime_to_datetime(str_time):
    return datetime.datetime.strptime(str_time, '%Y-%m-%d %H:%M:%S')

def divide_queries_based_on_time(tsv_stream):
    with MultipleOpenFiles() as files:
        for row in tqdm(tsv_stream):
            if row[2] == 'QueryTime':
                continue
            
            querytime = isotime_to_datetime(row[2])
            fileId = f'{querytime.month}_{querytime.day}'

            if not files.get(fileId):
                folder = f'data/dates/{fileId}'
                os.mkdir(folder)
                files.add(fileId, f'{folder}/queries')

            files.writeline(fileId, row[1])

def equality_divide_array(array, n_of_batches):
    segment_len = len(array) // n_of_batches

    for batch_id in range(0, n_of_batches):
        offset = batch_id * segment_len
        yield array[offset: offset + segment_len]

def main():
    archives = [
        download_and_save(url)
        for url in urls
    ]

    # vectorizer = TfidfVectorizer()
    # X = vectorizer.fit_transform(map(lambda row: row[1], get_text_from_gzip(archives)))
    # print(X)
    
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

    divide_queries_based_on_time(get_text_from_gzip(archives))
    
    path = f'data/dates/'
    days = os.listdir(path)
    day_batches = equality_divide_array(days, 2)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for batch in day_batches:
            futures.append(executor.submit(process, folders=batch, path=path))

        for future in concurrent.futures.as_completed(futures):
            print(future.result())


    # nlp = get_pipe()
    # tokenizer = get_tokenizer(nlp)

    
    # for folder in tqdm(os.listdir(path)):
    #     if folder not in sys.argv:


            # data, data_norm, queries, queries_norm = queries_to_vector(nlp, tokenizer, open(f'{path}{folder}/queries', 'r'))

            # if len(data) <= 1:
            #     continue

            # labels = vector_to_scatterplot(data, queries, folder)
            
            # if len(labels) == 0:
            #     continue
            
            # labels_norm = vector_to_scatterplot(data_norm, queries_norm, folder, sufix='_norm')
            # with open(f'{path}{folder}/cluster_similarity', 'w') as f:
            #     f.write(str(adjusted_rand_score(labels, labels_norm)))

def process(path, folders):
    nlp = get_pipe()
    tokenizer = get_tokenizer(nlp)

    for folder in folders:
        data, data_norm, queries, queries_norm = queries_to_vector(nlp, tokenizer, open(f'{path}{folder}/queries', 'r'))

        if len(data) <= 1:
            continue

        labels = vector_to_scatterplot(data, queries, folder)
        
        if len(labels) == 0:
            continue
        
        labels_norm = vector_to_scatterplot(data_norm, queries_norm, folder, sufix='_norm')
        with open(f'{path}{folder}/cluster_similarity', 'w') as f:
            f.write(str(adjusted_rand_score(labels, labels_norm)))

if __name__ == '__main__':
    main()
