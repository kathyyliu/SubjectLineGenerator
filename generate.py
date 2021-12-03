import re
import os
import json
import nltk
from datetime import datetime
import multiprocessing
from email.parser import BytesParser
from nltk.tokenize import word_tokenize, sent_tokenize

"""
Tokenizing emails for use in our Subject Line Generator.
    TO DO:
    - sent_tokenize instead of word_tokenize
    - ensure proper punctuation inclusion/exclusion
"""

Q = None
POOL = None


def tokenize(sentences=None, sentence=None):
    """Must call with either sentences, or sentence, not both"""
    final = []

    def tokenize_sentence(s1):
        """Sub-function to tokenize just one sentence"""
        p = re.compile(r'^[re:]|^[fwd:]|^=|[.edu]')
        q = re.compile(r'[^\w\s]')
        new_tokens1 = []

        # If not reply or forward
        if not p.match(s1):
            # Don't get rid of mailing list emails, just the "[...]" part
            if ']' in s1:
                try:
                    s1 = s1.split('] ')[1]
                except IndexError:
                    pass
            # Split into individual sentences
            new_sentences1 = s1.split('. ')
            for s in new_sentences1:
                tokens1 = word_tokenize(s)
                for tok in tokens1:
                    # Ignore special characters
                    if not q.match(tok) and len(tok) > 2:
                        new_tokens1.append(tok)
        return new_tokens1

    if sentences and not sentence:
        # If function was given list of many sentences
        for s1 in sentences:
            s = tokenize_sentence(s1)
            if s:
                final.append(s)
    elif sentence and not sentences:
        # If just one sentence was given
        s = tokenize_sentence(sentence)
        if s:
            final.append(s)
    else:
        # If incorrect input was given
        print("Must call tokenize() with either string or list of strings, not both")
    return final


def main():
    global POOL, Q
    path = './EmailSampleData/'
    final_json = {"emails": []}
    subject_lines = {}
    bodies = {}
    id = 0
    # tuples blueprint: ({[tokenized subject]}, [[list], [of], [tokenized], [body], [sentences]]])
    # Subject is always assumed to be just one sentence
    # "final_json['emails']" will be list of tuples, one per email

    for f in os.listdir(path):
        with open(path + f, 'rb') as file:
            # Parse subject lines
            header = BytesParser().parse(file)
            subject = header['subject']
            subject = subject.replace('\n\t', ' ').lower()
            subject_lines[f"{id}"] = subject

            # Very convoluted way to parse body text
            payload = header.get_payload()[0]
            if not isinstance(payload, str):
                message = payload.as_string()
            else:
                message = payload
            body = ''
            for line in message.split('\n'):
                # This is where blind trust comes into play
                skip = False
                if re.compile(r'^[--0]|^[Content\-Type]|=').match(line):
                    skip = True
                if re.compile(r'^<').match(line):
                    break
                if not skip:
                    body = body + f" {line}\n"
            bodies[f"{id}"] = body
        id += 1

    i = 0
    start = datetime.now()
    length = len(subject_lines.keys())
    for x in subject_lines.keys():
        # Generate the tokenized lists with multiprocessing
        a = subject_lines[f"{x}"]
        b = bodies[f"{x}"].split('. ')
        POOL.apply(t, (a, b))
        i += 1
        stats(length, start, i, 'Creating processes...     ')

    i = 0
    start = datetime.now()
    while True:
        output = Q.get()
        if output[0] is not None and output[1] is not None:
            final_json['emails'].append(output)
            i += 1
            stats(length, start, i, 'Finishing processes...      ')
        if i % 50 == 0:
            with open('data.json', 'w') as save_file:
                json.dump(final_json, save_file, indent=4)
        if i == length:
            break
    print(f'Data has been generated and saved')
    with open("data.json", "w") as save_file:
        json.dump(final_json, save_file, indent=4)
    return


def t(a, b):
    """Tokenize the subject (a) and the body (b), then put result in multiprocessing queue"""
    c = tokenize(sentence=a)
    d = tokenize(sentences=b)
    Q.put((c, d))


def stats(length, start, i, message):
    """Display process statistics."""
    ave_time = (datetime.now() - start).seconds / i
    minutes_left = int(((length - i) * ave_time) / 60)
    if minutes_left > 0:
        minutes_message = f'Approximately {minutes_left} minute(s) remaining'
    else:
        minutes_message = f'Less than a minute remaining                    '

    print(f'{message}\n'
          f'{round((i / length) * 100, 2)}% done     \n'
          f'{minutes_message}',
          end='\r\033[A\r\033[A\r')


def setup():
    """Initialize starting values and global variables."""
    global Q, POOL
    multiprocessing.set_start_method('fork')
    Q = multiprocessing.Manager().Queue()
    POOL = multiprocessing.Pool(6)
    nltk.download('averaged_perceptron_tagger')
    nltk.download('tagsets')


if __name__ == '__main__':
    setup()
    main()
