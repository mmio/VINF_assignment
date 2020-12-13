from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.decomposition import PCA

# def update_cluster(model, data):
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         return model.partial_fit(data)

# def cluster_data(data, e, s):
#     return Birch(n_clusters=500).fit(data)

# def reduce_dimensions(data, n_components):
    # # PCA
    # pca_of_n = TruncatedSVD(n_components)
    # return pca_of_n.fit_transform(data)

# def learn_tfidf(row_stream):
    # vectorizer = TfidfVectorizer()
    # return vectorizer.fit(
    #     map(
    #         lambda row: row[1],
    #         row_stream))

# def create_scatterplot(data, labels, base_path, sufix):
    #     print("tsne start")
    #     tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000, n_jobs=1)
    #     tsne_results = tsne.fit_transform(data)
    #     print("tsne finished saving")
    #     save_scatterplot(f'{base_path}{sufix}.pdf', tsne_results[:,0], tsne_results[:,1], labels) 

def save_scatterplot(savefile, x, y, hue):
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x=x, y=y, hue=hue
    )
    plt.savefig(savefile)