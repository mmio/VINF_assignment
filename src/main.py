import sys
import warnings
import os
import numpy as np
from tqdm import tqdm
import datetime
import multiprocessing
import itertools as it

import concurrent.futures
from collections import Counter
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

def queries_to_vector(nlp, tokenizer, row_stream):
    hot = HistogramOfQueries('data/global_stats/')
    query_stream = map(lambda query: query[:-1], row_stream)

    uniq_queries = set(query_stream)

    for doc in nlp.pipe(uniq_queries, disable=['ner', 'tagger']):
        ## Keep only english queries
        if doc._.language != 'en':
            continue

        normalized = ''
        without_stopwords = ''
        for token in doc:
            if not token.is_stop and not token.is_oov:
                normalized = f'{normalized} {token.lemma_}'
                without_stopwords = f'{without_stopwords} {token.text}'
        normalized = normalized.lstrip()
        without_stopwords = without_stopwords.lstrip()

        if len(normalized) == 0:
            continue

        ## collect stats for the day
        hot.add_doc(doc, 0)

        dq = tokenizer(without_stopwords)

        dqq = tokenizer(normalized)

        yield dq.vector, dqq.vector, dq.text, dqq.text

    hot.save()

def reduce_dimensions(data, n_components):
    pca_of_n = PCA(n_components)
    return pca_of_n.fit_transform(data)

def cluster_data(data, e, s):
    return Birch(n_clusters=500).fit(data)
    # return AgglomerativeClustering(
    #     n_clusters=None,
    #     linkage='ward',
    #     compute_full_tree=True,
    #     distance_threshold=e,
    #     memory=Memory('./cachedir', verbose=0)
    # ).fit(data)
    # return OPTICS(eps=e, min_samples=s, n_jobs=4).fit(data)
    # return DBSCAN(eps=0.01, min_samples=2, n_jobs=4).fit(data)
    # return KMeans(n_clusters=20).fit(data)

def save_scatterplot(savefile, x, y, hue):
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x=x, y=y, hue=hue
    )
    plt.savefig(savefile)

def update_cluster(model, data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.partial_fit(data)

def online_clustering(data_subset, model):

    # reduced_data = reduce_dimensions(data_subset, 50)
    reduced_data = data_subset
    
    if len(reduced_data) == 0:
        return

    update_cluster(model, reduced_data)

def divide_queries_based_on_time(tsv_stream):
    with MultipleOpenFiles() as files:
        for row in tqdm(tsv_stream):

            querytime = isotime_to_datetime(row[2])
            fileId = f'{querytime.month}_{querytime.day}'

            if not files.get(fileId):
                folder = f'data/dates/{fileId}'
                os.mkdir(folder)
                files.add(fileId, f'{folder}/queries')

            files.writeline(fileId, row[1])

## Util
def equality_divide_array(array, n_of_batches):
    segment_len = len(array) // n_of_batches

    for batch_id in range(0, n_of_batches):
        offset = batch_id * segment_len
        yield array[offset: offset + segment_len]

def isotime_to_datetime(str_time):
    return datetime.datetime.strptime(str_time, '%Y-%m-%d %H:%M:%S')

def iter_by_batch(iter, n):
    while True:
        acc = list(it.islice(iter, n))
        if len(acc) == 0:
            break
        yield acc

def main():
    archives = [
        download_and_save(url)
        for url in urls
    ]

    # vectorizer = TfidfVectorizer()
    # X = vectorizer.fit_transform(map(lambda row: row[1], get_text_from_gzip(archives)))
    # print(X)

    # textStatsCollectors = [
    #     HistogramOfQueries('data/users/global/stats/'),
    #     AverageQueryLength('data/users/global/stats/'),
    #     AverageNumberOfQueriesPerUser('data/users/global/stats/')
    # ]

    # docStatsCollectors = [
    #     HistogramOfTokens('data/users/global/stats/histogramOfTokens.json'),
    #     # DictionaryOfTokens('data/users/global/stats/dict.pickle'),
    # ]

    divide_queries_based_on_time(get_text_from_gzip(archives))
    
    path = f'data/dates/'
    days = os.listdir(path)
    day_batches = list(equality_divide_array(days, 1))
    
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     futures = []
    #     for batch in day_batches:
    #         futures.append(executor.submit(process, path=path, folders=batch))

    #     for future in concurrent.futures.as_completed(futures):
    #         print(future.result())

    jobs = []
    for batch in day_batches:
        p = multiprocessing.Process(target=process, args=(path, batch, ))
        jobs.append(p)
        p.start()


def process(path, folders):
    nlp = get_pipe()
    tokenizer = get_tokenizer(nlp)
    batch_size = 100

    for folder in folders:
        cl_model = Birch(n_clusters=300)
        cl_model_norm = Birch(n_clusters=300)

        # Memory?
        vectors = list(queries_to_vector(nlp, tokenizer, open(f'{path}{folder}/queries', 'r')))

        print(f'doing {folder}')
        for coll in iter_by_batch(iter(vectors), batch_size):
            unzipped = list(zip(*coll))

            data = np.array(unzipped[0])
            data_norm = np.array(unzipped[1])

            if len(data) <= 1:
                continue

            online_clustering(data, cl_model)
            online_clustering(data_norm, cl_model_norm)
        
        ls = []
        ls_norm = []
        data_for_sne = []
        data_norm_for_sne = []
        for coll in iter_by_batch(iter(vectors), batch_size):
            unzipped = list(zip(*coll))

            data = unzipped[0]
            data_norm = unzipped[1]
            queries = unzipped[2]
            queries_norm = unzipped[3]

            ls.extend(zip(cl_model.predict(data), queries))
            ls_norm.extend(zip(cl_model_norm.predict(data_norm), queries_norm))

            data_for_sne.extend(data)
            data_norm_for_sne.extend(data_norm)
        
        # print('writing')
        # with open(f'readme{folder}.txt', 'w') as f:
        #     for item in ls:
        #         f.write(f'{str(item)}\n')

        # with open(f'readme{folder}_norm.txt', 'w') as f:
        #     for item in ls_norm:
        #         f.write(f'{str(item)}\n')
                
        # compare clusters of normalized and non-normalized queries
        results = list(zip(*ls))
        labels = results[0]
        query_subset = results[1]

        results_norm = list(zip(*ls_norm))
        labels_norm = results_norm[0]
        query_subset_norm = results_norm[1]

        with open(f'{path}{folder}/cluster_similarity', 'w') as f:
            f.write(str(adjusted_rand_score(labels, labels_norm)))

        # save cluster queries
        os.mkdir(f'{path}{folder}/clusters')
        for cluster_id, query in ls:
            with open(f'{path}{folder}/clusters/{cluster_id}', 'a') as f:
                f.write(f'{query}\n')

        os.mkdir(f'{path}{folder}/clusters_norm')
        for cluster_id, query in ls_norm:
            with open(f'{path}{folder}/clusters_norm/{cluster_id}', 'a') as f:
                f.write(f'{query}\n')

        print("tsne start")
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000, n_jobs=1)
        tsne_results = tsne.fit_transform(data_for_sne)
        print("tsne finished")
        save_scatterplot(f'data/dates/{folder}/{folder}.pdf', tsne_results[:,0], tsne_results[:,1], labels)

        print("tsne start")
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000, n_jobs=1)
        tsne_results = tsne.fit_transform(data_norm_for_sne)
        print("tsne finished")
        save_scatterplot(f'data/dates/{folder}/{folder}_norm.pdf', tsne_results[:,0], tsne_results[:,1], labels_norm)

        # save cluster of vectors data to folder, for further comparison
        os.mkdir(f'{path}{folder}/cluster_dump')
        for label in set(labels):
            with open(f'{path}{folder}/cluster_dump/{label}', 'wb') as fh:
                for i in range(len(labels)):
                    if labels[i] == label:
                        pickle.dump(data_for_sne[i], fh)

        os.mkdir(f'{path}{folder}/cluster_dump_norm')
        for label in set(labels_norm):
            with open(f'{path}{folder}/cluster_dump_norm/{label}', 'wb') as fh:
                for i in range(len(labels_norm)):
                    if labels_norm[i] == label:
                        pickle.dump(data_norm_for_sne[i], fh)


if __name__ == '__main__':
    main()
