import os
import sys
import warnings
import pickle
import contextlib
import numpy as np
import itertools

import datetime
import multiprocessing
import itertools as it
import concurrent.futures

from tqdm import tqdm
from collections import Counter
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans, OPTICS, AgglomerativeClustering, Birch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

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

def get_query(tsv_stream):
    for row in tsv_stream:
        yield row[1]

def queries_to_vector(nlp, tokenizer, filename):
    # textStatsCollectors = [
    #     HistogramOfQueries('data/users/global/stats/'),
    #     AverageQueryLength('data/users/global/stats/'),
    #     AverageNumberOfQueriesPerUser('data/users/global/stats/')
    # ]

    # docStatsCollectors = [
    #     HistogramOfTokens('data/users/global/stats/histogramOfTokens.json'),
    #     # DictionaryOfTokens('data/users/global/stats/dict.pickle'),
    # ]

    with open(filename, 'r') as row_stream:
        # hot = HistogramOfQueries('data/global_stats/')
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
            # hot.add_doc(doc, 0)

            tokens = tokenizer(without_stopwords)

            tokens_normalized = tokenizer(normalized)

            yield tokens.vector, tokens_normalized.vector, tokens.text, tokens_normalized.text

        # hot.save()

def reduce_dimensions(data, n_components):
    # PCA
    pca_of_n = TruncatedSVD(n_components)
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
    update_cluster(model, data_subset)

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

def equality_divide_array(array, n_of_batches):
    segment_len = len(array) // n_of_batches

    for batch_id in range(0, n_of_batches + 1):
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

# def learn_tfidf(row_stream):
    # vectorizer = TfidfVectorizer()
    # return vectorizer.fit(
    #     map(
    #         lambda row: row[1],
    #         row_stream))

def learn_tfidf_2(stream):
    vectorizer = TfidfVectorizer(max_features=300)
    return vectorizer.fit(stream)

def compute_stats(n_proc, tfidf=None):
    path = f'data/dates/'
    days = os.listdir(path)
    day_batches = list(equality_divide_array(days, n_proc))

    jobs = []
    for batch in day_batches:
        p = multiprocessing.Process(target=process_folders, args=(path, batch, tfidf, ))
        jobs.append(p)
        p.start()

    for job in jobs:
        job.join()

def spacy_preprocess(n_proc):
    path = f'data/dates/'
    days = os.listdir(path)
    day_batches = list(equality_divide_array(days, n_proc))

    jobs = []
    for days in day_batches:
        p = multiprocessing.Process(target=preprocess_folders, args=(path, days, ))
        jobs.append(p)
        p.start()

    for job in jobs:
        job.join()

def main():
    # archives = [
    #     download_and_save(url)
    #     for url in urls
    # ]

    # divide_queries_based_on_time(get_text_from_gzip(archives))

    # spacy_preprocess(n_proc=7)

    path = 'data/dates/'

    files = [
        open(f'{path}{folder}/queries')
        for folder in os.listdir(path)
    ]

    tfidf = learn_tfidf_2(list(itertools.chain(*files)))

    compute_stats(n_proc=1, tfidf=tfidf)

    # index()

    # compute_stats_with_indexes()

def get_tfidf_rep(queries, dictionary):
    return dictionary.transform(queries).toarray()

def read_from_pickle(path):
    with open(path, 'rb') as file:
        try:
            while True:
                yield pickle.load(file)
        except EOFError:
            pass

def preprocess_folders(path, folders):
    nlp = get_pipe()
    tokenizer = get_tokenizer(nlp)

    ## Pre-process normalized
    for folder in folders:
        print(f'Started processing {folder}')
        with open(f'{path}{folder}/processed', 'wb') as proc_file:
            for coll in queries_to_vector(nlp, tokenizer, f'{path}{folder}/queries'):
                pickle.dump(coll, proc_file)
            print(f'Finished processing {folder}')

    print("Finished processing folders")

def process_folders(path, folders, tfidf_dict=None, additional_stats_collectors=None):
    batch_size = 100

    cmodels = [
        ('w2v', Birch(n_clusters=300),
            lambda uz: (np.array(uz[0]), uz[2])),
        ('w2v_n', Birch(n_clusters=300),
            lambda uz: (np.array(uz[1]), uz[3])),
        ('tfidf', Birch(n_clusters=300),
            lambda uz: (
                get_tfidf_rep(uz[2], tfidf_dict),
                uz[2])),
        ('tfidf_n', Birch(n_clusters=300),
            lambda uz: (
                get_tfidf_rep(uz[3], tfidf_dict),
                uz[3]))
    ]

    def train_model(path, cmodel, get_data_and_query):
        print(f'training model {path}')
        reader = read_from_pickle(f'{path}/processed')
        for coll in iter_by_batch(reader, batch_size):
            print(f'iterating {folder}')
            unzipped = list(zip(*coll))
            data, _ = get_data_and_query(unzipped)
            online_clustering(data, cmodel)

    def create_dir(path):
        if not os.path.exists(path):
            os.mkdir(path)

    def predict_with_model(path, name, cmodel, get_data_and_query):
        print(f'predicting with {name}')

        create_dir(f'{path}/{name}')

        with open(f'{path}/{name}/label_query_vector', 'wb') as label_query_vector:
            reader = read_from_pickle(f'{path}/processed')
            for coll in iter_by_batch(reader, batch_size):
                print(f'iterating ------ {folder}')
                unzipped = list(zip(*coll))
                data, queries = get_data_and_query(unzipped)
                for i in zip(cmodel.predict(data), queries, data):
                    pickle.dump(i, label_query_vector)

        create_dir(f'{path}/cluster_{name}')
        create_dir(f'{path}/cluster_{name}_dump')

        for label, query, vector in read_from_pickle(f'{path}/{name}/label_query_vector'):
            with open(f'{path}/cluster_{name}/{label}', 'a') as f:
                f.write(f'{query}\n')

            with open(f'{path}/cluster_{name}_dump/{label}', 'wb') as fh:
                pickle.dump(vector, fh)
    
    for folder in folders:
        labels = []

        for name, cmodel, get_data in cmodels:
            train_model(f'{path}{folder}', cmodel, get_data)
            predict_with_model(f'{path}{folder}', name, cmodel, get_data)

            labels.append([
                label
                for label, _, _
                in read_from_pickle(f'{path}{folder}/{name}/label_query_vector')
            ])

        # read labels, then compare
        with open(f'{path}{folder}/cluster_similarity', 'w') as sim_file:
            sim_file.write(str(adjusted_rand_score(labels[0], labels[1])))
            sim_file.write(str(adjusted_rand_score(labels[0], labels[2])))
            sim_file.write(str(adjusted_rand_score(labels[0], labels[3])))
            sim_file.write(str(adjusted_rand_score(labels[1], labels[2])))
            sim_file.write(str(adjusted_rand_score(labels[1], labels[3])))
            sim_file.write(str(adjusted_rand_score(labels[2], labels[3])))

        # def create_scatterplot(data, labels, base_path, sufix):
        #     print("tsne start")
        #     tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000, n_jobs=1)
        #     tsne_results = tsne.fit_transform(data)
        #     print("tsne finished saving")
        #     save_scatterplot(f'{base_path}{sufix}.pdf', tsne_results[:,0], tsne_results[:,1], labels) 

if __name__ == '__main__':
    main()
