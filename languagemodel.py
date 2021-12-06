import extractkeys
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.model_selection import train_test_split
from keytotext import trainer
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
        if len(tok_phrase) <= 1:
            continue
        x = extractkeys.head(tok_phrase, False)
        if not x or len(x) < 3:
            continue
        keywords = ''
        for token in x:
            keywords += str(token) + ' '
        detokenize = TreebankWordDetokenizer().detokenize
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

    # train_df = make_dataset('common_gen', split='train')
    # print(train_df)
    # test_df = make_dataset('common_gen', split='test')
    #
    # model = trainer()
    # model.from_pretrained(model_name="t5-small")
    # model.train(train_df=train_df[:100], test_df=test_df[:50], batch_size=2, max_epochs=3, use_gpu=False)

    train, test = train_test_split(labeled_data, shuffle=True)
    model = trainer()
    model.from_pretrained()
    model.train(train_df=train, test_df=test, use_gpu=False)
    model.save_model()
    return model


def generate(model):
    # generate subject line with model
    while True:
        x = input("Input the email body or 'exit' to exit:")
        if x == 'exit':
            break
        sents = sent_tokenize(x)
        if len(sents) < 5:
            print("Sorry! Email is too short.")
            continue
        body = []
        # sent and word tokenize input
        for sent in sents:
            body.append(word_tokenize(sent))
        # extract keywords from body
        keywords = []
        key_sents = extractkeys.embedding(body, False)
        for sent in key_sents:
            # print(f"representative sent: {detokenize(sent)}")
            phrase = extractkeys.rake(sent)[0]
            tok_phrase = word_tokenize(phrase)
            x = extractkeys.head(tok_phrase, False)
            if not x:
                continue
            for token in x:
                keywords.append(str(token))
        if len(keywords) < 3:
            print("Sorry! Try again")
            continue
        # generate
        subject = model.predict(keywords, use_gpu=False)
        print(f"Suggested subject line:\n{subject}")


def main():
    # os.environ["TOKENIZERS_PARALLELISM"] = "False"
    # detokenize = TreebankWordDetokenizer().detokenize
    # # read in all subject lines
    # f = open('smalldata.json', )
    # data = json.load(f)['emails']
    # subject_lines = []
    # for email in data:
    #     if email[0][0] and len(email[0][0]) > 4:
    #         subject_lines.append(email[0][0])
    # # label subject lines with their key words
    # df = label(subject_lines)
    # # train model
    # model = train(df)
    # generate(model)
    generate(' ')


if __name__ == '__main__':
    main()
