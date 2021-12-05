import re
import json
import nltk
import multiprocessing
from mailbox import mbox
from datetime import datetime
from nltk.tokenize import word_tokenize, sent_tokenize

"""
Tokenizing emails for use in our Subject Line Generator.
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
        print("\n\n\nError: Must call tokenize() with either string or list of strings, not both")
        print(f"{sentence}\n{sentences}")
        exit(1)
    return final


def get_charsets(msg):
    """Keep track of possible charsets used for encoding"""
    charsets = set({})
    for c in msg.get_charsets():
        if c is not None:
            charsets.update([c])
    return charsets


def get_body(msg):
    """Parse an email for its body text"""
    while msg.is_multipart():
        msg = msg.get_payload()[0]
    b = msg.get_payload(decode=True)
    for charset in get_charsets(msg):
        b = b.decode(charset)
    return b


def t(a, b):
    """Tokenize the subject (a) and the body (b), then put result in multiprocessing queue"""
    c = tokenize(sentence=a)
    d = tokenize(sentences=b)
    Q.put((c, d))


def stats(length, start, i, message):
    """Display runtime statistics"""
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
    """Initialize starting values and global variables"""
    global Q, POOL
    multiprocessing.set_start_method('fork')
    Q = multiprocessing.Manager().Queue()
    POOL = multiprocessing.Pool(6)
    nltk.download('averaged_perceptron_tagger')
    nltk.download('tagsets')


def main():
    global POOL, Q
    mb = mbox('emails.mbox')
    final_json = {"emails": []}
    subject_lines = {}
    bodies = {}
    id = 0
    # tuples blueprint: ({[tokenized subject]}, [[list], [of], [tokenized], [body], [sentences]]])
    # Subject is always assumed to be just one sentence
    # "final_json['emails']" will be list of tuples, one per email

    # Parse for subject lines and email bodies
    for message in mb:
        dmessage = dict(message.items())
        subject = dmessage['Subject']
        body = get_body(message)
        subject_lines[f"{id}"] = subject
        bodies[f"{id}"] = body
        id += 1

    i = 0
    start = datetime.now()
    length = len(subject_lines.keys())
    for x in subject_lines.keys():
        # Generate the tokenized lists with multiprocessing
        a = subject_lines[f"{x}"]
        b = []
        try:
            b = sent_tokenize(bodies[f"{x}"])
        except Exception:
            print(bodies[f"{x}"])
        new_b = []

        # Get rid of whitespace and random characters
        for m in b:
            new_m = re.sub(r"[\n]|[\s ]+", " ", m).lower()
            new_m = re.sub(r"[<]|[>]|[*]", "", new_m)
            new_b.append(new_m)

        # Stopgap measure for weird bug
        if not len(new_b) > 1:
            new_b.append(" ")

        POOL.apply(t, (a, new_b))
        i += 1
        stats(length, start, i, 'Creating processes...     ')

    i = 0
    start = datetime.now()
    while True:
        # Accrue output data and save periodically
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


if __name__ == '__main__':
    setup()
    main()
