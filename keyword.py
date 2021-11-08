from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import MiniBatchKMeans
import pandas as pd
import json


def embedding(data):
    tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(data)]
    # embed each sentence in email body
    model = Doc2Vec(tagged_data, vector_size=20, window=2, min_count=1, workers=4)
    k = len(data) // 2
    mb = 10 * k
    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=mb).fit(model.wv)
    df_clusters = pd.DataFrame({
        "tokens": [" ".join(sent) for sent in data],
        "cluster": kmeans.labels_
    })
    print(df_clusters)


def main():
    f = open('data.json', )
    data = json.load(f)['emails']
    print(data[1])
    embedding(data[1])

    # s1 = "Hey everyone, We’ll be hosting our second Super Smash Bros. " \
    #      "Ultimate tournament with OxySmash TOMORROW, November 6, 2021 from 1:30 - 6:30pm in Johnson 203 (2nd floor of Johnson Hall). " \
    #      "The bracket will begin at 2:00pm, and we'll have setups for warm up before! All levels of experience are welcome! " \
    #      "We’ll be streaming some of the games on Twitch, and there will be opportunities to commentate. " \
    #      "There will also be opportunities for casual games, so even if you’re not super competitive, " \
    #      "make sure to swing by and have some fun meeting new people! " \
    #      "Signups can be found here: https://smash.gg/tournament/oxysmash-monthly-tournament-2/details" \
    #      "(Sign ups are not mandatory to compete, but helps us get a sense of who’s coming) " \
    #      "We're also looking for additional monitors to help the tournament run quicker, so if you have one we can borrow please let us know!" \
    #      "Feel free to contact us at aprichett@oxy.edu or lchico@oxy.edu, or in the Oxy.GG discord server, linked below, if you have any questions" \
    #      "Thanks, and hope to see you there!"
    # # q = compile(r'[^\w\s]')
    # new_tokens1 = []
    # new_sentences1 = s1.split('. ')
    # for s in new_sentences1:
    #     tokens1 = word_tokenize(s)
    #     for tok in tokens1:
    #         # Ignore special characters
    #         # if not q.match(tok) and len(tok) > 2:
    #         new_tokens1.append(tok)
    # print(new_tokens1)


if __name__ == '__main__':
    main()