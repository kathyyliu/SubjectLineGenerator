import extractkeys
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from keytotext import trainer, pipeline
import pandas as pd
import json


def label(subject_lines):
    labeled = {
        "subject line": [],
        "keywords": []
    }
    # extract keys from subject lines as their labels
    for line in subject_lines:
        phrase = extractkeys.rake(line)[0]
        tok_phrase = word_tokenize(phrase)
        if len(tok_phrase) > 1:
            x = extractkeys.head(tok_phrase, False)
            if not x:
                continue
            keywords = [x]
        else:
            keywords = [tok_phrase[0]]
        labeled["subject line"].append(line)
        labeled["keywords"].append(keywords)
    df = pd.DataFrame(labeled)
    return df


def train(labeled_data):
    train, test = train_test_split(labeled_data, shuffle=True)
    model = trainer()
    model.from_pretrained()
    model.train(train_df=train, test_df=test, use_gpu=False)
    return model


def generate(model, keywords):
    pass


def main():
    f = open('data.json', )
    data = json.load(f)['emails']
    subject_lines = []
    for email in data:
        subject_lines.append(email[0][0])
    df = label(subject_lines)
    model = train(df)
    print(model.predict(["attend", "session"]))


if __name__ == '__main__':
    main()
