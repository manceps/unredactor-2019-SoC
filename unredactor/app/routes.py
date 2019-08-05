from flask import request
from flask import render_template
from app import app


def sort_words(text):
    return ' '.join(sorted(text.split()))


@app.route('/')
@app.route('/index')
def index():
    text = request.args.get('text')
    unredacted_text = sort_words(text)
    return render_template('index.html', text=text, unredacted_text=unredacted_text)


@app.route('/about')
def about():
    return render_template('about.html', **locals())
