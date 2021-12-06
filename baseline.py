import random
from nltk.tokenize import word_tokenize
import string


# generates 5-word subject line of random words from email
def generate_baseline(email):
    tokens = []
    for token in word_tokenize(email):
        if token not in string.punctuation:
            tokens.append(token.lower())
    subject_line = ''
    for i in range(5):
        x = random.randint(0, len(tokens))
        subject_line += str(tokens[x]) + ' '
    return subject_line


def main():
    email = ''
    with open("email.txt", 'r') as file:
        for line in file:
            email += line.strip() + ' '
    print(generate_baseline(email))
    # print(email)


if __name__ == '__main__':
    main()