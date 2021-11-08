import os
import json
# import nltk
from email.parser import BytesParser
from re import compile
from nltk.tokenize import word_tokenize


def tokenize(sentences=None, sentence=None):
    """Must call with either sentences, or sentence, not both"""
    final = []

    def tokenize_sentence(s1):
        """Sub-function to tokenize just one sentence"""
        p = compile(r'^[re:]|^[fwd:]|^=|[.edu]')
        q = compile(r'[^\w\s]')
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
                if compile(r'^[--0]|^[Content\-Type]|=').match(line):
                    skip = True
                if compile(r'^<').match(line):
                    break
                if not skip:
                    body = body + f" {line}\n"
            bodies[f"{id}"] = body
        id += 1

    for i in subject_lines.keys():
        # Generate the tokenized lists
        s = tokenize(sentence=subject_lines[f"{i}"])
        b = tokenize(sentences=bodies[f"{i}"].split('. '))
        if s and b:
            final_json['emails'].append((s, b))

    with open("data.json", "w") as save_file:
        # Save data
        json.dump(final_json,
                  save_file,
                  indent=4)


if __name__ == '__main__':
    # nltk.download('averaged_perceptron_tagger')
    # nltk.download('tagsets')
    main()
