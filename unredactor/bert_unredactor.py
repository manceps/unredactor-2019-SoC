# -*- coding: utf-8 -*-


import os
import codecs
import numpy as np
from keras_bert import load_trained_model_from_checkpoint, Tokenizer


MARKER = 'unk'         # our abbreviated UNKOWN word marker (blank)
MASK_TOKEN = '[MASK]'  # defined by the BERT model

BERT_MODEL_CASED = True #@param {type:"boolean"}

if BERT_MODEL_CASED:
  UNZIPPED_MODEL_PATH = "/opt/models/wwm_cased_L-24_H-1024_A-16"
else:
  UNZIPPED_MODEL_PATH = "/opt/models/wwm_uncased_L-24_H-1024_A-16"

CONFIG_PATH = "UNZIPPED_MODEL_PATH/bert_config.json"
CHECKPOINT_PATH = "UNZIPPED_MODEL_PATH/bert_model.ckpt"
DICT_PATH = "UNZIPPED_MODEL_PATH/vocab.txt"


global P
P = None


class NLPPipeline(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            setattr(self, k, v)

def load_pipeline(unzipped_model_path=UNZIPPED_MODEL_PATH, cased=BERT_MODEL_CASED):

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
    tokenizer = Tokenizer(token_dict, cased=cased)

    return NLPPipeline(model=model, token_dict=token_dict, token_dict_rev=token_dict_rev, tokenizer=tokenizer)

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
    >>> find_repeated_substring('unredact unk if you can.')  # FIXME: shoudl be [1, 4, 8]? Why?!
    [9]
    >>> find_repeated_substring(' unkunk unk ')
    [1, 4, 7]
    """
    substring = substring or MARKER
    start = text.find(substring)
    stop = start + len(substring)
    starts = []
    for i in range(max_occurences):
        #if not (start > -1 and stop <= len(text) - len(substring) + 1): # AK - adds no value but misses the case when marker is the last word
        #    break
        if len(starts):
            stop = starts[-1] + len(substring)
            starts.append(stop + start)
        else:
            starts = [start]
        start = text[stop:].find(substring)
        if start < 0 and len(starts) > 1:
            return starts[:-1]

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

    all_actual_tokens = []
    actual_tokens = actual_tokens or [MASK_TOKEN] * num_redactions
    k = 0
    for i, masked_tok in enumerate(tokens_original):
        if i - 1 in redactions:
            all_actual_tokens.append(actual_tokens[k])
            k += 1
        else:
            all_actual_tokens.append(masked_tok)
    #print(f'    Actual: {[tok for (i, tok) in enumerate(all_actual_tokens) if i - 1 in redactions]}')

    return (predictions_parameterized, tokens)

def unredact(text, marker=MARKER, redacted_tokens=None):
    global P
    if not P:
        P = load_pipeline()
    unredacted = ' '
    marker = marker or 'unk'

    redactions = find_repeated_substring(text, substring=marker)
    if not redactions:
        print('No redactions found')
        unredacted = text
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

    j = 0
    count_correct = 0
    for (i, tok) in enumerate(all_tokens):
        if tok == '[MASK]' and j < len(unredacted_tokens):
            all_tokens[i] = unredacted_tokens[j]
            if redacted_tokens:
                count_correct += int(unredacted_tokens[j] == redacted_tokens[j])
            j += 1
    if redacted_tokens and redacted_tokens[0] != MASK_TOKEN:
      redaction_count = len(unredacted_tokens)

    unredacted_text = ' '.join(all_tokens)

    return unredacted_text, all_tokens, unredacted_tokens, count_correct, redaction_count

"""## Test 1"""

# input_text = "Anyone who stops learning is old, whether at twenty or eighty."

# redacted_text = "Anyone who stops learning is unk, whether at twenty or unk."

# redacted_tokens = ['old','eighty']

# unredacted_text, all_tokens, unredacted_tokens, count_correct, redaction_count = unredact(redacted_text, redacted_tokens=redacted_tokens)

# print(unredacted_text)
# print(all_tokens)
# print(unredacted_tokens)
# print(f' {count_correct} out of {redaction_count} redacted tokens were correctly predicted by BERT.')


def unredact_interactively():
    global P
    if not P:
        P = load_pipeline()
    unredacted = ' '
    while unredacted:
        text = input('Text: ')
        #marker = input('Redaction marker: ')
        #marker = marker or 'unk'
        marker = 'unk'
        redactions = find_repeated_substring(text, substring=marker)
        if not redactions:
            print('No redactions found')
            unredacted = text
            continue
        # print(redactions)
        start, stop = redactions[0], redactions[-1] + len(marker)
        prefix, suffix = text[:start], text[stop:]
        # print(start, stop)
        # print(f'prefix: {prefix}')
        # print(f'suffix: {suffix}')
        prefix_tokens = P.tokenizer.tokenize(prefix)[:-1]
        suffix_tokens = P.tokenizer.tokenize(suffix)[1:]
        # print(f'prefix_tokens: {prefix_tokens}')
        # print(f'suffix_tokens: {suffix_tokens}')
        unredacted_tokens, all_tokens = unredact_tokens(prefix_tokens=prefix_tokens, suffix_tokens=suffix_tokens, num_redactions=len(redactions))
        print(f'all_tokens: {all_tokens}')
        print(f'unredacted_tokens: {unredacted_tokens}')
        j = 0
        for (i, tok) in enumerate(all_tokens):
            if tok == '[MASK]' and j < len(unredacted_tokens):
                all_tokens[i] = unredacted_tokens[j]
                j += 1

        unredacted = ' '.join(all_tokens)
        # unredacted = ' '.join([t[2:] if t.startswith('##') else t for t in unredacted_tokens])
        print(f'Unredacted text: {unredacted}')

if __name__ == '__main__':
    unredact_interactively()
