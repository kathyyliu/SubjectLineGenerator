from keytotext import trainer, make_dataset
import json


def train(subject_lines):



def main():
    f = open('data.json', )
    data = json.load(f)['emails']
    subjectlines = []
    for email in data:
        subjectlines.append(email[0][0])


if __name__ == '__main__':
    main()
