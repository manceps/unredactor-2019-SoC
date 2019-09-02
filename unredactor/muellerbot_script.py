# import os
import codecs
import sys
import re
# import flask
import numpy as np
import pandas as pd
# import scipy
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
# import tensorflow as tf

"""### Global variables

Let's specify some global variables to hold things like paths.
Globals in python are always written in all caps.
That makes it easier to find them and replace them with local variables later.
Local variables make your code more modular, easier to reuse and maintain.
"""

MARKER = 'unk'         # our abbreviated UNKOWN word marker (blank)
MASK_TOKEN = '[MASK]'  # defined by the BERT model

BERT_MODEL_CASED = False
BERT_MODELS_DIR = "~/apps/unredactor/unredactor/app/models/uncased_L-12_H-768_A-12"

BERT_MODEL_DATE = "2018_10_18"
BERT_MODEL_NAME = "uncased_L-12_H-768_A-12"

BERT_MODEL_DIR = "$BERT_MODELS_DIR/$BERT_MODEL_NAME"
BERT_MODEL_ZIP = "$BERT_MODEL_DIR.zip"
UNZIPPED_MODEL_PATH = "~/apps/unredactor/unredactor/app/models/uncased_L-12_H-768_A-12"
CONFIG_PATH = "$UNZIPPED_MODEL_PATH/bert_config.json"
CHECKPOINT_PATH = "$UNZIPPED_MODEL_PATH/bert_model.ckpt"
DICT_PATH = "$UNZIPPED_MODEL_PATH/vocab.txt"

global P
P = None

"""## UNNECESSARY CODE

These are functions we used to process the Mueller report. You may not ever need them.
But there here in case you wan to learn more about regular expressions.

You can use this function to find redactions surrounded by square brackets, if you like.
But we'll just use the "unk" marker for now. So you don't even have to run this cell.
"""


def find_redactions(s):
    return re.findall(r'\[[^\]]*\]', s)


"""We used this function to try to find recactions markers in the Mueller Report.
The OCR usually contained redactions marked with "[HARM]", or "[Personal Privacy]" or
"[Grand Jury]", so this function returns a list of some bits of text that look like that.
"""


def get_probable_redactions(df):
    possible_redactions = set()
    for s in df.Text:
        possible_redactions = possible_redactions.union(set(find_redactions(s)))
    probable_redactions = set()
    for r in probable_redactions:
        if 'grand jury' in r.lower() or 'harm' in r.lower() or 'priva' in r.lower():
            probable_redactions.add(r)
    return probable_redactions


def guess_redaction_markers():
    df = pd.read_csv('mueller-report-factbase-with-redactions-marked.csv', header=1)
    r = get_probable_redactions(df)
    print(r)
    print()
    print(REDACTION_MARKERS)


def normalize_redaction_markers(lines, inplace=True):
    normalized_lines = [''] * len(lines) if not inplace else lines
    normalizer = dict(zip(REDACTION_MARKERS, ['__' + x.replace(' ', '_').replace('-', '_')[1:-1] + '__' for x in REDACTION_MARKERS]))
    for i, line in enumerate(lines):
        for k, v in normalizer.items():
            normalized_lines[i] = line.replace(k, v)
    return lines


def clean_dataframe(filepath='mueller-report-with-redactions-marked.csv'):
    df = pd.read_csv(filepath, header=1)
    df.columns = 'page text appendix unnamed'.split()
    return df


def get_line_pairs(df, redaction_marker='[Harm to Ongoing Matter]',
                   min_line_length=40, max_line_length=120):
    line_context = get_line_context(
        df=df, redaction_marker=redaction_marker,
        min_line_length=min_line_length, max_line_length=min_line_length)
    return [tuple(lc[:2]) for lc in line_context]


def get_line_context(df, redaction_marker='[Harm to Ongoing Matter]',
                     min_line_length=40, max_line_length=120):
    line_pairs = []
    for i, line in enumerate(df.text):
        if redaction_marker in line:
            prevline = df.text.iloc[i - 1]
            nextline = df.text.iloc[i + 1] if (i < (len(df) - 1)) else ''
            if (redaction_marker not in prevline and
                    len(prevline) > min_line_length and
                    len(prevline) < max_line_length):
                line_pairs.append((prevline, line, nextline))
    return line_pairs


def find_text(df='mueller-report-with-redactions-marked.csv', substring='of documents and', marker='[Personal Privacy]'):
    df = clean_dataframe(df) if isinstance(df, str) else df
    text = ''
    for t in df.text:
        if substring in t:
            text = t
            break
    # print(f'TEXT: {text}')
    marker_start = text.find(marker)
    marker_stop = marker_start + len(marker)
    # print(f'marker_start: {marker_start}, marker_stop: {marker_stop}')
    prefix = text[:marker_start]
    suffix = text[marker_stop:]
    # print(f'HOM prefix: {prefix}\nHOM suffix: {suffix}')

    return prefix, suffix


REDACTION_MARKERS = set([
    '[Harm to Ongoing Matter - Grand Jury]',
    '[Harm to Ongoing Matter - Personal Privacy]',
    '[Harm to Ongoing Matter]',
    '[Personal Privacy - Grand Jury]',
    '[Personal Privacy]',
    '[HOM]',
    '[Investigative Technique]',
    '[Investigative Technique]',
    '[IT]',
    'unk'
])

MARKER = 'unk'         # our short marker to redact text
MASK_TOKEN = '[MASK]'  # the MARKER token used in BERT to identify a token to predict


REDACTION_MARKERS = [
    '[Harm to Ongoing Matter - Grand Jury]',
    '[Harm to Ongoing Matter - Personal Privacy]',
    '[Harm to Ongoing Matter]',
    '[Personal Privacy - Grand Jury]',
    '[Personal Privacy]',
    '[HOM]',
]


def get_unredacted_sentences(df='mueller-report-with-redactions-marked.csv',
                             min_line_length=60, max_line_length=150):
    """ Find contiguous sections of unredacted text in the Mueller report to use in fine-tuning
    BERT."""
    df = clean_dataframe(df) if isinstance(df, str) else df
    line_pairs = get_line_pairs(df, min_line_length=min_line_length, max_line_length=max_line_length)
    print(pd.DataFrame(line_pairs, columns='text redacted_text'.split()).head())

    for i, (text, redacted) in enumerate(line_pairs):
        if len(re.findall(r'^[-:.0-9 \t]{1,2}', text.strip())) > 0:
            print(f'Skipping: {text[:30]}')
            continue
        print(f"Redacting: {text[:30]}")
        print()


def find_first_hom_tokens(df, text=None, substring='of documents and', marker='[Personal Privacy]'):
    global P
    P = P or load_pipeline()
    if not text:
        df = clean_dataframe(df) if isinstance(df, str) else df
        for t in df.text:
            if substring in t:
                text = t
                break
    # print(f'TEXT: {text}')
    tokens = P.tokenizer.tokenize(text)
    joined_tokens = ' '.join(tokens)
    # print(f'joined_tokens: {joined_tokens}')
    hom = ' '.join(P.tokenizer.tokenize(marker)[1:-1])
    # print(f'joined_hom: {hom}')
    hom_start = joined_tokens.find(hom)
    hom_stop = hom_start + len(hom)
    # print(f'hom_start: {hom_start}, hom_stop: {hom_stop}')
    prefix_tokens = joined_tokens[:hom_start].split()
    suffix_tokens = joined_tokens[hom_stop:].split()
    # print(f'HOM prefix_tokens: {prefix_tokens}\nHOM suffix_tokens: {suffix_tokens}')

    return prefix_tokens, suffix_tokens


sentences = [
    'The IRA later used social media accounts and interest groups to sow discord in the U.S. political system through what it termed "information warfare."',
    ('The campaign evolved from a generalized program designed in 2014 and 2015 to undermine the U.S. electoral system,' +
        'to a targeted operation that by early 2016 favored candidate Trump and disparaged candidate Clinton.'),
    ('The IRA\'s operation also included the purchase of political advertisements on social media in the names of U.S. persons and entities,' +
        'as well as the staging of political rallies inside the United States.'),
    ('To organize those rallies, IRA employees posed as U.S. grassroots entities and persons and made contact' +
        ' with Trump supporters and Trump Campaign officials in the United States.'),
    ]


def unredact_text(text, redactions=[2, 3]):
    print(f"Redacting tokens {redactions} in: {text}")
    global P
    P = P or load_pipeline()

    tokens = P.tokenizer.tokenize(text)
    tokens_original = tokens.copy()
    for r in redactions:
        tokens[r + 1] = MASK_TOKEN

    # print(f'Tokens: {tokens}')

    indices = np.asarray([[P.token_dict[token] for token in tokens] + [0] * (512 - len(tokens))])
    segments = np.asarray([[0] * len(tokens) + [0] * (512 - len(tokens))])
    masks = np.asarray([[0] * 512])
    for r in redactions:
        masks[0][r + 1] = 1
    # masks = np.asarray([[0, 1, 1] + [0] * (512 - 3)])

    predicts = P.model.predict([indices, segments, masks])[0]
    predicts = np.argmax(predicts, axis=-1)
    predictions_parameterized = list(
        map(lambda x: P.token_dict_rev[x],
            [x for (j, x) in enumerate(predicts[0]) if j - 1 in redactions])
        )
    # predictions_hardcoded = list(map(lambda x: token_dict_rev[x], predicts[0][3:5]))
    print(f'Predictions: {" ".join(predictions_parameterized)}')

    # print(f'Hardcoded fill with: {predictions_hardcoded}')
    # list(map(lambda x: token_dict_rev[x], predicts[0][1:3]))
    print(f'.    Actual: {[t for (i, t) in enumerate(tokens_original) if i - 1 in redactions]}')
    print()
    print()
    # if len(predictions) > 10:
    #     break
    return (predictions_parameterized, text)


"""## The Bot

This section of code uses the keras-bert model to try to guess any of the words in some text you input into the bot.

This class allows us to put all our steps in our NLP pipeline in one place. It really just converts a dictionary of key value pairs:

    {'key1": "value1", ... "key99": "value99"}

Into an object with attributes for each key-value pair:

    >>> pipe = NLPPipeline({'key1": "value1", "key99": "value99"}
    >>> pipe.key1
    'value1'
    >>> pipe.key99
    'value99'

We'll use it to store things like the tokenizer and word vector embedder and the BERT model itself.
"""


class NLPPipeline(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            setattr(self, k, v)


"""This is where we load the keras-bert model into an attribute of global variable called P that holds our entire NLP pipeline."""


def load_pipeline(unzipped_model_path=UNZIPPED_MODEL_PATH, cased=BERT_MODEL_CASED):
    if len(sys.argv) != 4:
        print('python load_model.py CONFIG_PATH CHECKPOINT_PATH DICT_PATH')
        print('CONFIG_PATH:     UNZIPPED_MODEL_PATH/bert_config.json')
        print('CHECKPOINT_PATH: UNZIPPED_MODEL_PATH/bert_model.ckpt')
        print('DICT_PATH:       UNZIPPED_MODEL_PATH/vocab.txt')

    config_path, checkpoint_path, dict_path = (unzipped_model_path + '/bert_config.json',
                                               unzipped_model_path + '/bert_model.ckpt',
                                               unzipped_model_path + '/vocab.txt')

    model = load_trained_model_from_checkpoint(config_path, checkpoint_path, training=True)
    model.summary(line_length=120)

    token_dict = {}
    with codecs.open(dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    token_dict_rev = {v: k for k, v in token_dict.items()}
    if cased:
        print('***************CASED TOKENIZER*******************')
    else:
        print('***************uncased tokenizer*******************')
    tokenizer = Tokenizer(token_dict, cased=cased)

    return NLPPipeline(model=model, token_dict=token_dict, token_dict_rev=token_dict_rev, tokenizer=tokenizer)


"""### Load the pipeline

Now we'll load the pipeline using the pretrained BERT model as our language model.
This is the interesting part, where all the many layers of the BERT model are revealed as Keras layers.
"""

P = load_pipeline()


def find_repeated_substring(text, substring=MARKER, max_occurences=32):
    """ Find contiguous redaction markers and return the start locations

    >>> text = 'Mueller said "MASK MASK MASK", then walked away.'
    >>> find_repeated_substring(text, 'MASK')
    [14, 19, 24]
    >>> find_repeated_substring('unkunkunk')
    [0, 3, 6]
    >>> find_repeated_substring(' unkunkunk')
    [1, 4, 7]
    >>> find_repeated_substring(' unkunkunk ')
    [1, 4, 7]
    >>> find_repeated_substring('unredact unk if you can.')  # FIXME: shoudl be [1, 4, 8]?
    [9]
    >>> find_repeated_substring(' unkunk unk ')  # FIXME: shoudl be [1, 4, 8]?
    [1, 4, 8]
    """
    # print(f'TEXT: {text}')
    substring = substring or MARKER
    start = text.find(substring)
    stop = start + len(substring)
    starts = []
    for i in range(max_occurences):
        if not (start > -1 and stop <= len(text) - len(substring) + 1):
            break
        # print(start, stop)
        if len(starts):
            stop = starts[-1] + len(substring)
            starts.append(stop + start)
        else:
            starts = [start]
        # print(start, stop)
        start = text[stop:].find(substring)
        if start < 0 and len(starts) > 1:
            return starts[:-1]
        # print(start, stop)
        # print(starts)
    return starts


def unredact_tokens(prefix_tokens=[], suffix_tokens=[], num_redactions=5, actual_tokens=None):
    global P
    if not P:
        P = load_pipeline()
    tokens = list(prefix_tokens) + [MASK_TOKEN] * num_redactions + list(suffix_tokens)
    tokens = tokens[:512]
    tokens_original = tokens.copy()

    indices = np.asarray([[P.token_dict[token] for token in tokens] + [0] * (512 - len(tokens))])
    segments = np.asarray([[0] * len(tokens) + [0] * (512 - len(tokens))])
    masks = np.asarray([[0] * 512])
    redactions = []
    for i, t in enumerate(tokens):
        if t == MASK_TOKEN:
            redactions.append(i - 1)
            masks[0][i] = 1

    predicts = P.model.predict([indices, segments, masks])[0]
    predicts = np.argmax(predicts, axis=-1)
    predictions_parameterized = list(
        map(lambda x: P.token_dict_rev[x],
            [x for (j, x) in enumerate(predicts[0]) if j - 1 in redactions])
        )
    print(f'Predictions: {predictions_parameterized}')

    all_actual_tokens = []
    actual_tokens = actual_tokens or [MASK_TOKEN] * num_redactions
    k = 0
    for i, masked_tok in enumerate(tokens_original):
        if i - 1 in redactions:
            all_actual_tokens.append(actual_tokens[k])
            k += 1
        else:
            all_actual_tokens.append(masked_tok)
    print(f'    Actual: {[tok for (i, tok) in enumerate(all_actual_tokens) if i - 1 in redactions]}')

    return (predictions_parameterized, tokens)


def unredact_bert(text, marker=MARKER, redacted_tokens=None):
    global P
    if not P:
        P = load_pipeline()
    marker = marker or 'unk'

    redactions = find_repeated_substring(text, substring=marker)
    if not redactions:
        print('No redactions found')
        return redactions

    start, stop = redactions[0], redactions[-1] + len(marker)
    prefix, suffix = text[:start], text[stop:]
    prefix_tokens = P.tokenizer.tokenize(prefix)[:-1]
    suffix_tokens = P.tokenizer.tokenize(suffix)[1:]
    unredacted_tokens, all_tokens = unredact_tokens(
        prefix_tokens=prefix_tokens,
        suffix_tokens=suffix_tokens,
        num_redactions=len(redactions),
        actual_tokens=redacted_tokens)

    print(f'all_tokens: {all_tokens}')
    print(f'unredacted_tokens: {unredacted_tokens}')

    j = 0
    count_correct = 0
    for (i, tok) in enumerate(all_tokens):
        if tok == '[MASK]' and j < len(unredacted_tokens):
            all_tokens[i] = unredacted_tokens[j]
            if redacted_tokens:
                count_correct += int(unredacted_tokens[j] == redacted_tokens[j])
            j += 1
    if redacted_tokens and redacted_tokens[0] != MASK_TOKEN:
        print(f' {count_correct} out of {len(unredacted_tokens)} redacted tokens were correctly predicted by BERT.')

    unredacted_text = ' '.join(all_tokens)

    return unredacted_text


unredacted_text = unredact_bert("To be or not to unk, that is the question.")

print(unredacted_text)

unredacted_text = unredact_bert("To be or not to unk, that is the question.", redacted_tokens=['be'])

print(unredacted_text)

unredacted_text = unredact_bert("Is the president of the United States named unk unk ?", redacted_tokens=['Barak', 'Obama'])

unredacted_text = unredact_bert("And now I'd like to introduce you to the 2008 President of the United States of America, unk unk !",
                                redacted_tokens=['Barak', 'Obama'])

"""## Cased BERT Model
So the uncased BERT model did fine on sentences where case isn't important.
But let's see if it can do better on proper nouns (famous people names).
We'll need to load the "cased" BERT model for that.
"""

MARKER = 'unk'

BERT_MODEL_CASED = True
BERT_MODELS_DIR = "~/apps/unredactor/unredactor/app/models/cased_L-12_H-768_A-12"

BERT_MODEL_DATE = "2018_10_18"
BERT_MODEL_NAME = "cased_L-12_H-768_A-12"

BERT_MODEL_DIR = "$BERT_MODELS_DIR/$BERT_MODEL_NAME"
BERT_MODEL_ZIP = "$BERT_MODEL_DIR.zip"
UNZIPPED_MODEL_PATH = "~/apps/unredactor/unredactor/app/models/cased_L-12_H-768_A-12"
CONFIG_PATH = "$UNZIPPED_MODEL_PATH/bert_config.json"
CHECKPOINT_PATH = "$UNZIPPED_MODEL_PATH/bert_model.ckpt"
DICT_PATH = "$UNZIPPED_MODEL_PATH/vocab.txt"


P = load_pipeline(unzipped_model_path=UNZIPPED_MODEL_PATH, cased=BERT_MODEL_CASED)

unredacted_text = unredact_bert("And now I'd like to introduce you to the 2008 President of the United States of America, unk unk !",
                                redacted_tokens=['Barak', 'Obama'])

unredacted_text = unredact_bert("The President of the United States of America after George W Bush was unk unk .", redacted_tokens=['Barak', 'Obama'])

unredact_bert("The President of the United States, unk , made an announcement today about the Iraq war.", redacted_tokens=['Bush'])


"""Still no joy. We'll have to train it on some text with the names in it that matter to us, if we want it to know some important facts about them."""
