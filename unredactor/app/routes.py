import csv,datetime
from flask import render_template, request
from flask import jsonify
# from flask import redirect, flash, url_for

from app.constants import context
from app import app

# from app.nlp import sort_words
# from muellerbot import unredact_bert
from unredactor_functions import sort_and_replace_unks, unredact_text_get_and_words

from app.forms import UnredactForm


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', **context)


@app.route('/about')
def about():
    return render_template('about.html', **context)


@app.route('/api/unredact_bert')
def api_unredact_bert():
    text = request.args.get('text')
    get_words = request.args.get('get_words', "True")
    if isinstance(get_words, str):
        get_words = 'true' in get_words.lower() or get_words in '1TtYy'
    unredacted_text = ""
    unredacted_words = []
    unredacted_text, unredacted_words = unredact_text_get_and_words(text, get_words=get_words)
    # unredacted_words = list(unredacted_text.split())
    with open('/home/msoc/api_queries.csv', mode='a') as aq:
        aq_writer = csv.writer(aq, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
        aq_writer.writerow([text,unredacted_text,unredacted_words,str(datetime.datetime.now()),request.remote_addr])
    return jsonify(dict(text=text, unredacted_text=unredacted_text, unredacted_words=unredacted_words))
    # return render_template('unredacted.json', text=text, unredacted_text=unredacted_text, unredacted_words=unredacted_words)


@app.route('/api/sort_words')
def api_sort_words():
    unredacted_text, unredacted_words = None, None
    text = request.args.get('text')
    if text:
        unredacted_text, unredacted_words = sort_and_replace_unks(text, get_words=True)
    return jsonify(dict(text=text, unredacted_text=unredacted_text, unredacted_words=unredacted_words))
    # return render_template('unredacted.json', text=text, unredacted_text=unredacted_text, unredacted_words=unredacted_words)


@app.route('/unredactor', methods=['GET', 'POST'])
def unredactor():
    form = UnredactForm()
    text = ''
    unredacted_text = ""
    unredacted_words = []
    if form.validate_on_submit():
        text = str(form.text.data)
        unredacted_text = unredact_text_get_and_words(text)
    with open('/home/msoc/form_queries.csv', mode='a') as fq:
        fq_writer = csv.writer(fq, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
        fq_writer.writerow([text,unredacted_text,unredacted_words,str(datetime.datetime.now()),request.remote_addr])

    return render_template('unredact.html', title='Unredact', form=form, text=text, unredacted_text=unredacted_text, unredacted_words=unredacted_words)
