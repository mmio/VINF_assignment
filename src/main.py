import os
import sys
import warnings
import pickle
import numpy as np
import itertools

import datetime
import itertools as it
import concurrent.futures

from tqdm import tqdm
from sklearn.cluster import Birch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer

import multiprocessing
from multiprocessing import Pool, Process

from files import urls, destination
from download import download_and_save
from prepare import get_text_from_gzip
from pipeline import get_pipe, get_tokenizer

from processors.histogramOfTokens import HistogramOfTokens
from processors.histogramOfQueries import HistogramOfQueries
from MultipleOpenFiles import MultipleOpenFiles
from utils import read_from_pickle, equality_divide_array, isotime_to_datetime, iter_by_batch, create_dir

def queries_to_vector(nlp, tokenizer, filename):
    with open(filename, 'r') as row_stream:
        query_stream = map(lambda query: query[:-1], row_stream)

        uniq_queries = set(query_stream)

        for doc in nlp.pipe(uniq_queries, disable=['ner']):
            if doc._.language != 'en':
                continue

            normalized = ''
            vecs_normalized = []

            text = ''
            vecs = []
            for token in doc:
                if not token.is_stop and not token.is_oov:
                    normalized = f'{normalized} {token.lemma_}'.lstrip()
                    vecs_normalized.append(token.vocab[token.lemma].vector)

                    text = f'{text} {token.text}'.lstrip()
                    vecs.append(token.vector)

            if len(normalized) == 0 or len(text) == 0:
                continue

            yield np.mean(vecs, axis=0), np.mean(vecs_normalized, axis=0), text, normalized

def online_clustering(data, model):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.partial_fit(data)

def divide_queries_based_on_time(tsv_stream):
    with MultipleOpenFiles() as files:
        for row in tqdm(tsv_stream):

            querytime = isotime_to_datetime(row[2])
            fileId = f'{querytime.month}_{querytime.day}'

            if not files.get(fileId):
                folder = f'../data/dates/{fileId}'
                create_dir(folder)
                files.add(fileId, f'{folder}/queries')

            files.writeline(fileId, row[1])

def learn_tfidf(stream):
    vectorizer = TfidfVectorizer(max_features=300)
    return vectorizer.fit(stream)

def compute_stats(n_proc, tfidf=None):
    path = '../data/dates/'

    files = [
        open(f'{path}{folder}/queries')
        for folder in os.listdir(path)
    ]

    tfidf = learn_tfidf(list(itertools.chain(*files)))

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
    path = f'../data/dates/'
    days = os.listdir(path)
    day_batches = list(equality_divide_array(days, n_proc))

    jobs = []
    for days in day_batches:
        p = multiprocessing.Process(target=preprocess_folders, args=(path, days, ))
        jobs.append(p)
        p.start()

    for job in jobs:
        job.join()

def collect_global_stats():
    path = f'../data/dates/'
    files = [
        map(lambda x: x[2], read_from_pickle(f'{path}{folder}/processed'))
        for folder in os.listdir(path)
    ]

    files_n = [
        map(lambda x: x[3], read_from_pickle(f'{path}{folder}/processed'))
        for folder in os.listdir(path)
    ]

    hoq = HistogramOfQueries('../data/global_stats/hoq')
    hot = HistogramOfTokens('../data/global_stats/hot')
    for query in tqdm(itertools.chain(*files)):
        hoq.add_doc(query)
        hot.add_doc(query)
    hoq.save()
    hot.save()

    hoq = HistogramOfQueries('../data/global_stats/hoq_n')
    hot = HistogramOfTokens('../data/global_stats/hot_n')
    for query in tqdm(itertools.chain(*files_n)):
        hoq.add_doc(query)
        hot.add_doc(query)
    hoq.save()
    hot.save()

def get_tfidf_rep(queries, dictionary):
    return dictionary.transform(queries).toarray()

def preprocess_folders(path, folders):
    nlp = get_pipe()
    tokenizer = get_tokenizer(nlp)

    print('Processing folders')
    for folder in tqdm(folders):
        with open(f'{path}{folder}/processed', 'wb') as proc_file:
            for coll in queries_to_vector(nlp, tokenizer, f'{path}{folder}/queries'):
                pickle.dump(coll, proc_file)

def process_folders(path, folders, tfidf_dict=None):
    batch_size = 100

    def train_model(path, cmodel, get_data_and_query):
        print(f'training model {path}')
        reader = read_from_pickle(f'{path}/processed')
        for coll in iter_by_batch(reader, batch_size):
            unzipped = list(zip(*coll))
            data, _ = get_data_and_query(unzipped)

            online_clustering(data, cmodel)

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

            with open(f'{path}/cluster_{name}_dump/{label}', 'ab') as fh:
                pickle.dump(vector, fh)

        coll = list(read_from_pickle(f'{path}/{name}/label_query_vector'))
        unzipped = list(zip(*coll))
        labels = unzipped[0]
        vectors = unzipped[2]

        with open(f'{path}/{name}/silhouette', 'w') as f:
            f.write(str(silhouette_score(vectors, labels)))
    
    for folder in folders:
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
            sim_file.write('w2v/w2v_n ' + str(adjusted_rand_score(labels[0], labels[1])) + '\n')
            sim_file.write('w2v/tfidf ' + str(adjusted_rand_score(labels[0], labels[2])) + '\n')
            sim_file.write('w2v/tfidf_n ' + str(adjusted_rand_score(labels[0], labels[3])) + '\n')
            sim_file.write('w2v_n/tfidf ' + str(adjusted_rand_score(labels[1], labels[2])) + '\n')
            sim_file.write('w2v_n/tfidf_n ' + str(adjusted_rand_score(labels[1], labels[3])) + '\n')
            sim_file.write('tfidf/tfidf_n ' + str(adjusted_rand_score(labels[2], labels[3])) + '\n')

def main():
    create_dir(f'../data')
    create_dir(f'../data/datasets')
    create_dir(f'../data/datasets/aol')
    create_dir(f'../data/dates')
    create_dir(f'../data/indices')
    create_dir(f'../data/global_stats')

    archives = [
        download_and_save(url)
        for url in urls
    ]

    divide_queries_based_on_time(get_text_from_gzip(archives))

    spacy_preprocess(n_proc=3)

    collect_global_stats()

    # cluster data
    compute_stats(n_proc=3)

if __name__ == '__main__':
    main()
