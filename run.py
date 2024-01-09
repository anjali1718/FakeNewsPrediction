from flask import Flask, render_template, redirect, url_for, flash
from forms import UserForm
from System.code import predict_fake_news, model, vectorizer, get_accuracy

app = Flask(__name__)
app.config['SECRET_KEY'] = 'b47206eae80abc5d27acdc2d7cee576e'

@app.route('/', methods=['GET', 'POST'])
def home():
    form = UserForm()
    accuracy = get_accuracy()
    if form.validate_on_submit():
        news = form.news.data
        title = form.title.data
        author = form.author.data
        user_input = author + ' ' + title + ' ' + news
        prediction = predict_fake_news(user_input, model, vectorizer)
        if(prediction != [0]): 
            flash('This news is real!', 'success')
        else:
            flash('This news is fake!', 'danger')
        return redirect(url_for('home'))
    return render_template('forms.html',  title='Prediction', form=form, accuracy=accuracy)

if __name__ == "__main__":
    app.run(debug=True)