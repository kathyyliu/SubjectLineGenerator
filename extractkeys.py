from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import pandas as pd
import numpy as np
import json


# https://dylancastillo.co/nlp-snippets-cluster-documents-using-word2vec/#how-to-cluster-documents
def embedding(data, verbose=False):
    tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(data)]
    # embed each sentence in email body
    model = Doc2Vec(tagged_data, vector_size=20, window=2, min_count=1, workers=4)
    # get list of embeddings
    vectors = []
    for i in range(len(tagged_data)):
        vectors.append(model.dv[i])
    # k = len(data) // 2      # num clusters TODO: find optimal k for given email length
    k = 4
    mb = 10 * k             # num mini batches
    # cluster embeddings with k-means
    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=mb).fit(vectors)
    avg_score = silhouette_score(vectors, kmeans.labels_)
    if verbose:
        print(f"\nFor n_clusters = {k}")
        print(f"Silhouette coefficient: {avg_score:0.2f}")      # positive/higher coef means better clusters
        df_clusters = pd.DataFrame({
            "tokens": [" ".join(sent) for sent in data],
            "cluster": kmeans.labels_
        })
        print(df_clusters)
    # calculate silhouette coefficient of each cluster
    sample_silhouette_values = silhouette_samples(vectors, kmeans.labels_)
    silhouette_values = []
    for i in range(k):
        cluster_silhouette_values = sample_silhouette_values[kmeans.labels_ == i]
        silhouette_values.append(
            (i,
            cluster_silhouette_values.shape[0],
            cluster_silhouette_values.mean(),)
        )
    # sort clusters by silhouette coefficient
    silhouette_values = sorted(
        silhouette_values, key=lambda tup: tup[2], reverse=True
    )
    # FIXME: whyyyy are the coefficients different between runs??
    if verbose:
        print(f"\nSilhouette values:")
        for s in silhouette_values:
            print(f"    Cluster {s[0]}: Size:{s[1]} | Avg:{s[2]:.2f}")
        # print("\nMost representative terms per cluster (based on centroids):")
    # find most representative tokens for each cluster
    # for i in range(k):
    #     tokens_per_cluster = ""
    #     most_representative = model.wv.most_similar(positive=[kmeans.cluster_centers_[i]], topn=5)
    #     for t in most_representative:
    #         tokens_per_cluster += f"{t[0]} "
    #     if verbose:
    #         print(f"Cluster {i}: {tokens_per_cluster}")
    # Find most representative sentence for all "good enough" clusters
    if verbose:
        print("\nKey sentences found:")
    key_sentences = []
    for s in silhouette_values:
        if s[2] >= avg_score/1.33 and s[2] > 0:     # my def of "good enough"
            cluster = s[0]
            most_representative_docs = np.argsort(
                np.linalg.norm(vectors - kmeans.cluster_centers_[cluster], axis=1)
            )
            d = most_representative_docs[0]
            if verbose:
                print(data[d])
            key_sentences.append(data[d])
    return key_sentences


def main():
    f = open('data.json', )
    data = json.load(f)['emails']
    print(data[0][1])
    embedding(data[0][1], True)


if __name__ == '__main__':
    main()