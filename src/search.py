import os
import sys
import pickle
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import lucene
 
from java.io import File
from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.index import IndexReader, DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.util import Version

import matplotlib.pyplot as plt
from pipeline import get_pipe, get_tokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def read_from_pickle(path):
    with open(path, 'rb') as file:
        try:
            while True:
                yield pickle.load(file)
        except EOFError:
            pass

if __name__ == "__main__":
    lucene.initVM()
    analyzer = StandardAnalyzer()

    subfolder_index_pairs = [
       ('cluster_w2v', 'index_w2v'),
       ('cluster_w2v_n', 'index_w2v_n'),
        ('cluster_tfidf', 'index_tfidf'),
        ('cluster_tfidf_n', 'index_tfidf_n')
    ]

    search_term = sys.argv[1]

    for cluster_type, index in subfolder_index_pairs:
        print(f'searching in {index}')
        path = Paths.get(f'../data/indices/{index}')
        
        reader = DirectoryReader.open(SimpleFSDirectory(path))
        searcher = IndexSearcher(reader)
 
        query = QueryParser("content", analyzer).parse(search_term)
        MAX = 1000000
        hits = searcher.search(query, MAX)
 
        month_counter = [
            [0] * 32,
            [0] * 32,
            [0] * 32
        ]

        for hit in hits.scoreDocs:
            doc = searcher.doc(hit.doc)

            month = int(doc.get('day').split('_')[0])
            day = int(doc.get('day').split('_')[1])

            month_counter[month-3][day] += 1

            cluster = doc.get('cluster')

        flat_months = [item for sublist in month_counter for item in sublist]
        fig, ax = plt.subplots(figsize=(30,8))
        b1 = ax.bar(range(len(flat_months)), flat_months)

        labels = []
        for month in [3,4,5]:
            for day in range(32):
                labels.append(f'{month}_{day}')

        plt.xticks(range(len(flat_months)), (labels), rotation=90, fontsize='xx-small')
        plt.savefig('search_result.pdf', bbox_inches='tight')
        plt.close()


        nlp = get_pipe()
        tokenizer = get_tokenizer(nlp)

        path = '../data/dates/'

        files = [
            open(f'{path}{folder}/queries')
            for folder in os.listdir(path)
        ]
        def learn_tfidf_2(stream):
            vectorizer = TfidfVectorizer(max_features=300)
            return vectorizer.fit(stream)
        tfidf = learn_tfidf_2(list(itertools.chain(*files)))

        avg_sims = []
        top_results = []
        # calculate similarity to cluster
        print("Calculating similarities")
        for hit in tqdm(hits.scoreDocs):
            # print(hit.score, hit.doc, hit.toString())
            doc = searcher.doc(hit.doc)

            day = doc.get('day')
            cluster = doc.get('cluster')

            search_term_vec = None
            vectors = []
            for i in read_from_pickle(f'../data/dates/{day}/{cluster_type}_dump/{cluster}'):
                i = i[:300]
                if len(i) < 300:
                    vectors.append(np.zeros(300))
                else:
                    vectors.append(i)

            if cluster_type in ['cluster_w2v', 'cluster_w2v_n']:
                search_term_vec = tokenizer(search_term).vector.reshape(1, -1)
            else:
                search_term_vec = tfidf.transform([search_term])

            similarities = cosine_similarity(search_term_vec, vectors)[0]
            avg_sims.append(np.mean(similarities))

            with open(f'../data/dates/{day}/{cluster_type}/{cluster}') as f:
                file_content = f.read().splitlines()
                n = 5
                if len(similarities) < 5:
                    n = len(similarities)
                results = np.argpartition(similarities, -n)[-n:]
                for ind in results:
                    if cluster_type in ['cluster_w2v']:
                        if file_content[ind-1] == search_term:
                            continue
                        top_results.append((file_content[ind-1], similarities[ind]))
                    elif cluster_type in ['cluster_w2v_n']:
                        if file_content[ind] == search_term:
                            continue
                        top_results.append((file_content[ind], similarities[ind]))
                    else:
                        if file_content[ind] == search_term:
                            continue
                        top_results.append((file_content[ind], similarities[ind]))
        for i in sorted(set(top_results), key=lambda x: x[1], reverse=True)[:10]:
            print(i[0])
        print(np.mean(avg_sims))
