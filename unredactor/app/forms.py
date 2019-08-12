from flask_wtf import FlaskForm
from wtforms import TextAreaField, SubmitField
from wtforms.validators import DataRequired

class UnredactForm(FlaskForm):
	text = TextAreaField('', validators=[DataRequired()])
	submit = SubmitField('Unredact')
