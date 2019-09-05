from flask import render_template, request
# from flask import redirect, flash, url_for

from app.constants import context
from app import app

# from app.nlp import sort_words
# from muellerbot import unredact_bert
from unredactor_functions import sort_and_replace_unks, unredact_text_v2

from app.forms import UnredactForm


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', **context)


@app.route('/about')
def about():
    return render_template('about.html', **context)


@app.route('/api/unredact-bert')
def api():
    unredacted_text, unredacted_words = None, None
    text = request.args.get('text')
    if text:
        unredacted_text, unredacted_words = text, ['word1', 'word2']
        # unredacted_text, unredacted_words = unredact_bert(text, get_words=True)
    return render_template('unredacted.json', text=text, unredacted_text=unredacted_text, unredacted_words=unredacted_words)

    # original_text = request.args.get('text')
    # print(len(original_text), type(original_text))
    # if original_text:
    #     unredacted_text, unredacted_words = unredact(original_text, get_words=True)
    # #    final_words = "["
    # #    for word in unredacted_words:
    # #        final_words = final_words + '"' + word + '", '
    # #    final_words = final_words[:len(final_words)-2] + "]"
    # #    unredacted_words = final_words
    # return render_template('unredacted.json', text=original_text, unredacted_text=unredacted_text, unredacted_words=unredacted_words)


@app.route('/api/sort_words')
def api_sort_words():
    unredacted_text, unredacted_words = None, None
    text = request.args.get('text')
    if text:
        unredacted_text, unredacted_words = sort_and_replace_unks(text, get_words=True)
    return render_template('unredacted.json', text=text, unredacted_text=unredacted_text, unredacted_words=unredacted_words)


@app.route('/unredactor', methods=['GET', 'POST'])
def unredactor():
    form = UnredactForm()
    text = ''

    # Depending on which module is imported, either the sort function (unredactor functions) or
    # the actual unredact function (muellerbot) runs

    unredacted_text, unredacted_words = '', []
    if form.validate_on_submit():
        text = str(form.text.data)
        unredacted_text, unredacted_words = unredact_text_v2(text, get_words=True)
    return render_template('unredact.html', title='Unredact', form=form, text=text, unredacted_text=unredacted_text, unredacted_words=unredacted_words)

