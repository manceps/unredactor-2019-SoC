from flask import request
from flask import render_template
from app import app

@app.route('/')
@app.route('/index')
def index():
	text = request.args.get('text')
	return render_template('unredacted.json', text=text)