from flask import render_template, flash, redirect, request, url_for
from app import app
from app.forms import UnredactForm
from unredactor_functions import unredact

def sort_words(text):
    return ' '.join(sorted(text.split()))


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', **locals())


@app.route('/about')
def about():
    return render_template('about.html', **locals())


#@app.route('/unredact')
#def unredact():
#    text = request.args.get('text')
#    unredacted_text = sort_words(text)
#    return render_template('unredacted_text.json', text=text, unredacted_text=unredacted_text)

@app.route('/unredactor', methods=['GET', 'POST'])
def unredactor():
	form = UnredactForm()
	unredacted_text = ''
	if form.validate_on_submit():
		flash('Unredact request')
		unredacted_text = unredact(form.text.data)
	return render_template('unredact.html', title='Unredact', form=form, unredacted_text=unredacted_text)
