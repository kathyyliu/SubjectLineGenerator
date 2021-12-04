from keytotext import trainer, make_dataset
import pandas
import json


def train(subject_lines):
    df = pandas.DataFrame(subject_lines)
    model = trainer()
    model.from_pretrained()
    model.train(train_df=df)



def main():
    f = open('data.json', )
    data = json.load(f)['emails']
    subjectlines = []
    for email in data:
        subjectlines.append(email[0][0])



if __name__ == '__main__':
    main()
