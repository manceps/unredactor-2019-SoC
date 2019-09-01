from flask import render_template, flash, redirect, request, url_for
from app import app

from app.constants import context
from app.nlp import sort_words
from app.forms import UnredactForm
from muellerbot import unredact
#from unredactor_functions import unredact

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', **context)


@app.route('/about')
def about():
    return render_template('about.html', **context)

@app.route('/api')
def api():
    unredacted_text, unredacted_words = None, None
    original_text = request.args.get('text')
    print(len(original_text), type(original_text))
    if original_text:
        unredacted_text, unredacted_words = unredact(original_text, get_words=True)
    #    final_words = "["
    #    for word in unredacted_words:
    #        final_words = final_words + '"' + word + '", '
    #    final_words = final_words[:len(final_words)-2] + "]"
    #    unredacted_words = final_words
    return render_template('unredacted.json', text=original_text, unredacted_text=unredacted_text, unredacted_words=unredacted_words)

@app.route('/api/sort_words')
def api_sort_words():
    unredacted_text, unredacted_words = None, None
    text = request.args.get('text')
    if text:
        unredacted_text, unredacted_words = unredact(text, get_words=True)
    return render_template('unredacted.json', text=text, unredacted_text=unredacted_text, unredacted_words=unredacted_words)


@app.route('/unredactor', methods=['GET', 'POST'])
def unredactor():
    form = UnredactForm()
    text = ''

    #Depending on which module is imported, either the sort function (unredactor functions) or
    #the actual unredact function (muellerbot) runs	

    if form.validate_on_submit():
         text = str(form.text.data)
    return render_template('unredact.html', title='Unredact', form=form, text=text, unredacted_text=unredact(text, True))

