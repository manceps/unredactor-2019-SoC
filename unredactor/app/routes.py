from flask import request
from flask import render_template, flash, redirect
from app import app
from app.forms import UnredactForm

def sort_words(text):
    return ' '.join(sorted(text.split()))


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', **locals())


@app.route('/about')
def about():
    return render_template('about.html', **locals())


@app.route('/unredact')
def unredact():
    text = request.args.get('text')
    unredacted_text = sort_words(text)
    return render_template('unredacted_text.json', text=text, unredacted_text=unredacted_text)

@app.route('/unredactor', methods=['GET', 'POST'])
def unredactor():
	form = UnredactForm()
	if form.validate_on_submit():
		flash('Unredact request')
		return redirect('/unredactor')
	return render_template('unredact.html', title='Unredact', form=form)
