import os
import random
from load_and_predict import unredact_text_get_and_words  # noqa


# TODO move this to constants and move words file to data/words.txt
word_dir = os.path.realpath(os.path.dirname(__file__))  # FIXED: dir->word_dir: don't use builtin function names as variable names
filename = os.path.join(word_dir, 'app', 'words')  # FIXED: don't use path sep ("/" or "\\"), os.path.join uses sep for your OS (Win/Linux)
words = open(filename).read().splitlines()


def sort_and_replace_unks(text, get_words=False):
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
