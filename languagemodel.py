import extractkeys
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.model_selection import train_test_split
from keytotext import trainer
import pandas as pd


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
    train, test = train_test_split(labeled_data, shuffle=True)
    model = trainer()
    model.from_pretrained()
    model.train(train_df=train, test_df=test, use_gpu=False)
    model.save_model()
    return model


def generate():
    # generate subject line with model
    while True:
        i = ""
        x = input("Input the email body or 'exit' to exit:")
        if x == 'exit':
            break
        try:
            with open(f"{x}", 'r') as file:
                i = file.read()
        except Exception as e:
            print("Nope, try again!")
            exit(1)
        sents = sent_tokenize(i)
        body = []
        # sent and word tokenize input
        for sent in sents:
            body.append(word_tokenize(sent))
        # extract keywords from body
        keywords = []
        key_sents = extractkeys.embedding(body, False)
        for sent in key_sents:
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
        return keywords


def main():
    t = trainer()
    t.load_model(model_dir='./model')
    keywords = generate()
    subject = t.predict(keywords, use_gpu=False)
    print(f"Suggested subject line:\n{subject}")


if __name__ == '__main__':
    main()
