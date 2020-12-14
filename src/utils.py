from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import itertools as it
import datetime

def read_from_pickle(path):
    with open(path, 'rb') as file:
        try:
            while True:
                yield pickle.load(file)
        except EOFError:
            pass

def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

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

def reduce_dimensions(data, n_components):
    pca_of_n = PCA(n_components)
    return pca_of_n.fit_transform(data)

def create_scatterplot(data, labels, base_path, sufix):
        print("tsne start")
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000, n_jobs=1)
        tsne_results = tsne.fit_transform(data)
        print("tsne finished saving")
        save_scatterplot(f'{base_path}{sufix}.pdf', tsne_results[:,0], tsne_results[:,1], labels) 

def save_scatterplot(savefile, x, y, hue):
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x=x, y=y, hue=hue
    )
    plt.savefig(savefile)