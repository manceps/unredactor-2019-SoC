import os
import random

dir = os.path.realpath('.')
filename = os.path.join(dir, 'app/words')
words = open(filename).read().splitlines()


def unredact(text, get_words=False):
    sorted_text = text
    listed_text = sorted_text.split()
    listed_text.sort()
    if get_words:
        random_list = []
        for word in listed_text:
            if word == 'unk':
                random_list.append(words[random.randint(1, len(words))])
        return (' '.join(listed_text), random_list)
    return ' '.join(listed_text)
