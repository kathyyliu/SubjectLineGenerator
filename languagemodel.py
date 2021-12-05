import extractkeys
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.model_selection import train_test_split
from keytotext import trainer, pipeline, make_dataset
import pandas as pd
import json
import os


def label(subject_lines):
    labeled = {
        "text": [],
        "keywords": []
    }
    # extract keys from subject lines as their labels
    for line in subject_lines:
        if len(line) <= 1:
            continue
        phrase = extractkeys.rake(line)[0]
        tok_phrase = word_tokenize(phrase)
        if len(tok_phrase) > 1:
            x = extractkeys.head(tok_phrase, False)
            if not x:
                continue
            keywords = str(x[0]) + ' ' + str(x[1])
        else:
            keywords = str(tok_phrase[0])
        detokenize = TreebankWordDetokenizer().detokenize
        print(line)
        print(keywords)
        labeled["text"].append(detokenize(line))
        labeled["keywords"].append(keywords)
    df = pd.DataFrame(labeled)
    print(df)
    return df


def train(labeled_data):
    # from keytotext import pipeline
    # nlp = pipeline("k2t-base")  # loading the pre-trained model
    # params = {"do_sample": True, "num_beams": 4, "no_repeat_ngram_size": 3, "early_stopping": True}  # decoding params
    # print(nlp(['Delhi', 'India', 'capital'], **params))  # keywords
    # return nlp

    train_df = make_dataset('common_gen', split='train')
    print(train_df)
    test_df = make_dataset('common_gen', split='test')

    model = trainer()
    model.from_pretrained(model_name="t5-small")
    model.train(train_df=train_df[:100], test_df=test_df[:50], batch_size=2, max_epochs=3, use_gpu=False)
    model.save_model()

    # train, test = train_test_split(labeled_data, shuffle=True)
    # model = trainer()
    # model.from_pretrained()
    # model.train(train_df=train, test_df=test, use_gpu=False)
    # return model


def generate(model, keywords):
    pass


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "False"

    # f = open('data.json', )
    # data = json.load(f)['emails']
    # subject_lines = []
    # for email in data:
    #     subject_lines.append(email[0][0])
    # df = label(subject_lines)
    df = 'm'
    train(df)
    # print(model.predict(["attend", "session"]))


if __name__ == '__main__':
    main()
