import os
import sys
import warnings
import pickle
import contextlib
import numpy as np

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
    # TruncatedSVD
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

    reduced_data = data_subset

    if data_subset.shape[1] > 300:
        reduced_data = reduce_dimensions(data_subset, 300)

    # reduced_data = reduce_dimensions(data_subset, 300)

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
    vectorizer = TfidfVectorizer()
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
    archives = [
        download_and_save(url)
        for url in urls
    ]

    divide_queries_based_on_time(get_text_from_gzip(archives))

    spacy_preprocess(n_proc=7)

    # compute_stats(n_proc=7)

    # index()

    # compute_stats_with_indexes()

def get_tfidf_rep(queries, dictionary):
    return dictionary.transform(queries)

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
    batch_size = 500

    cmodels = [
        ('w2v', Birch(n_clusters=300),
            lambda uz: (np.array(uz[0]), uz[2])),
        ('w2v_n', Birch(n_clusters=300),
            lambda uz: (np.array(uz[1]), uz[3]))
        # ('tfidf', Birch(n_clusters=300),
        #     lambda uz: (
        #         reduce_dimensions(get_tfidf_rep(uz[2], tfidf_dict), 300),
        #         uz[2])),
        # ('tfidf_n', Birch(n_clusters=300),
        #     lambda uz: (
        #         reduce_dimensions(get_tfidf_rep(uz[3], tfidf_dict), 300),
        #         uz[3]))
    ]

    def train_model(path, cmodel, get_data_and_query):
        print(f'training model {path}')
        reader = read_from_pickle(f'{path}/processed')
        for coll in iter_by_batch(reader, batch_size):
            print(f'iterating {folder}')
            unzipped = list(zip(*coll))
            data, _ = get_data_and_query(unzipped)
            online_clustering(data, cmodel)

    def predict_with_model(path, name, cmodel, get_data_and_query):
        print(f'predicting with {name}')

        label_query_pairs = []
        vectors = []

        reader = read_from_pickle(f'{path}/processed')
        for coll in iter_by_batch(reader, batch_size):
            print(f'iterating ------ {folder}')
            unzipped = list(zip(*coll))
            data, queries = get_data_and_query(unzipped)
            label_query_pairs.extend(zip(cmodel.predict(data), queries))
            vectors.extend(data)

        if not os.path.exists(f'{path}/cluster_{name}'):
            os.mkdir(f'{path}/cluster_{name}')
        for cluster_id, query in label_query_pairs:
            with open(f'{path}/cluster_{name}/{cluster_id}', 'a') as f:
                f.write(f'{query}\n')

        results = list(zip(*label_query_pairs))
        labels = results[0]

        if not os.path.exists(f'{path}/cluster_{name}_dump'):
            os.mkdir(f'{path}/cluster_{name}_dump')
        for label in set(labels):
            with open(f'{path}/cluster_{name}_dump/{label}', 'wb') as fh:
                for i in range(len(labels)):
                    if labels[i] == label:
                        pickle.dump(vectors[i], fh)
        
        return labels

    
    for folder in folders:
        labels = []

        for name, cmodel, get_data in cmodels:
            train_model(f'{path}{folder}', cmodel, get_data)
            labels.append(predict_with_model(f'{path}{folder}', name, cmodel, get_data))


        with open(f'{path}{folder}/cluster_similarity', 'w') as sim_file:
            sim_file.write(str(adjusted_rand_score(labels[0], labels[1])))

    # for folder in folders:
    #     print("Learning tfidf of the day")
    #     tfidf_dict = learn_tfidf_2(map(lambda x: x[2], read_from_pickle(f'{path}{folder}/processed')))

    #     # cl_model = Birch(n_clusters=300)
    #     # cl_model_norm = Birch(n_clusters=300)
    #     cl_model_tfidf = Birch(n_clusters=300)
    #     cl_model_norm_tfidf = Birch(n_clusters=300)

    #     print(f'doing {folder}')
    #     for coll in iter_by_batch(read_from_pickle(f'{path}{folder}/processed'), batch_size):
    #         print(f'iterating {folder}')
    #         unzipped = list(zip(*coll))

    #         # data = np.array(unzipped[0])
    #         # data_norm = np.array(unzipped[1])
    #         queries = unzipped[2]
    #         queries_norm = unzipped[3]

    #         tfidf = get_tfidf_rep(queries, tfidf_dict)
    #         tfidf_norm = get_tfidf_rep(queries_norm, tfidf_dict)

    #         # if len(data) <= 1:
    #         #     continue

    #         # online_clustering(data, cl_model)
    #         # online_clustering(data_norm, cl_model_norm)
    #         online_clustering(tfidf, cl_model_tfidf)
    #         online_clustering(tfidf_norm, cl_model_norm_tfidf)
        
    #     # ls = []
    #     # ls_norm = []
    #     ls_tfidf = []
    #     ls_norm_tfidf = []

    #     # vec_for_sne = []
    #     # vec_norm_for_sne = []
    #     vec_tfidf = []
    #     vec_tfidf_norm = []


    #     for coll in iter_by_batch(read_from_pickle(f'{path}{folder}/processed'), batch_size):
    #         print(f'iterating 2 {folder}')
    #         unzipped = list(zip(*coll))

    #         # data = unzipped[0]
    #         # data_norm = unzipped[1]

    #         queries = unzipped[2]
    #         queries_norm = unzipped[3]

    #         tfidf = reduce_dimensions(get_tfidf_rep(queries, tfidf_dict), 300)
    #         tfidf_norm = reduce_dimensions(get_tfidf_rep(queries_norm, tfidf_dict), 300)

    #         # ls.extend(zip(cl_model.predict(data), queries))
    #         # ls_norm.extend(zip(cl_model_norm.predict(data_norm), queries_norm))
    #         ls_tfidf.extend(zip(cl_model_tfidf.predict(tfidf), queries))
    #         ls_norm_tfidf.extend(zip(cl_model_norm_tfidf.predict(tfidf_norm), queries_norm))

    #         # vec_for_sne.extend(data)
    #         # vec_norm_for_sne.extend(data_norm)
    #         vec_tfidf.extend(tfidf)
    #         vec_tfidf_norm.extend(tfidf_norm)
                
        # compare clusters of normalized and non-normalized queries
        # results = list(zip(*ls_tfidf))
        # labels = results[0]

        # results_norm = list(zip(*ls_norm_tfidf))
        # labels_norm = results_norm[0]

        # with open(f'{path}{folder}/cluster_similarity_tfidf', 'w') as f:
        #     f.write(str(adjusted_rand_score(labels, labels_norm)))

        # # save cluster queries
        # os.mkdir(f'{path}{folder}/clusters_tfidf')
        # for cluster_id, query in ls_tfidf:
        #     with open(f'{path}{folder}/clusters_tfidf/{cluster_id}', 'a') as f:
        #         f.write(f'{query}\n')

        # os.mkdir(f'{path}{folder}/clusters_tfidf_norm')
        # for cluster_id, query in ls_norm_tfidf:
        #     with open(f'{path}{folder}/clusters_tfidf_norm/{cluster_id}', 'a') as f:
        #         f.write(f'{query}\n')

        # def create_scatterplot(data, labels, base_path, sufix):
        #     print("tsne start")
        #     tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000, n_jobs=1)
        #     tsne_results = tsne.fit_transform(data)
        #     print("tsne finished saving")
        #     save_scatterplot(f'{base_path}{sufix}.pdf', tsne_results[:,0], tsne_results[:,1], labels) 

        # vector_name_pairs = [
        #     (vec_tfidf, 'tfidf'),
        #     (vec_tfidf_norm, 'tfidf-norm'),
        #     (vec_for_sne, labels, 'non-norm'),
        #     (vec_norm_for_sne, labels_norm, 'norm')
        # ]

        # for data, sufix in vector_name_pairs:
        #     create_scatterplot(
        #         reduce_dimensions(data, 2),
        #         f'data/dates/{folder}/{folder}_',
        #         sufix)

        # save cluster of vectors data to folder, for further comparison
        # os.mkdir(f'{path}{folder}/cluster_dump_tfidf')
        # for label in set(labels):
        #     with open(f'{path}{folder}/cluster_dump_tfidf/{label}', 'wb') as fh:
        #         for i in range(len(labels)):
        #             if labels[i] == label:
        #                 pickle.dump(vec_tfidf[i], fh)

        # os.mkdir(f'{path}{folder}/cluster_dump_norm_tfidf')
        # for label in set(labels_norm):
        #     with open(f'{path}{folder}/cluster_dump_norm_tfidf/{label}', 'wb') as fh:
        #         for i in range(len(labels_norm)):
        #             if labels_norm[i] == label:
        #                 pickle.dump(vec_tfidf_norm[i], fh)

if __name__ == '__main__':
    main()
