from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import pandas as pd
import json


def embedding(data):
    tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(data)]
    print(tagged_data)
    # embed each sentence in email body
    model = Doc2Vec(tagged_data, vector_size=20, window=2, min_count=1, workers=4)
    # get list of embeddings
    vectors = []
    for i in range(len(tagged_data)):
        vectors.append(model.dv[i])
    k = len(data) // 2      # num clusters
    mb = 10 * k             # num mini batches
    # cluster embeddings with k-means
    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=mb).fit(vectors)
    df_clusters = pd.DataFrame({
        "tokens": [" ".join(sent) for sent in data],
        "cluster": kmeans.labels_
    })
    print(df_clusters)


def main():
    f = open('data.json', )
    data = json.load(f)['emails']
    print(data[0][1])
    embedding(data[0][1])


if __name__ == '__main__':
    main()