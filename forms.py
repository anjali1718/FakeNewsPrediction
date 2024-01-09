from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

class UserForm(FlaskForm):
    news = StringField('NEWS', validators=[DataRequired()])
    title = StringField('TITLE', validators=[DataRequired()])
    author = StringField('AUTHOR', validators=[DataRequired()])
    submit = SubmitField('submit')