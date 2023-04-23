#Implement all this concept by machine learning with flask

from flask import Flask, escape, request, render_template
import pickle
from nltk.corpus import stopwords
import string
import nltk
# nltk.download()
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

vector = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("finalized_model.pkl", 'rb'))

tfidf = pickle.load(open('vectorizer_e.pkl', 'rb'))
email_model = pickle.load(open('Email_Model.pkl', 'rb'))

app = Flask(__name__)

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)



@app.route('/')
def home():
    return render_template("index.html")

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == "POST":
        news = str(request.form['news'])
        # print(news)

        predict = model.predict(vector.transform([news]))[0]
        # print(predict)

        return render_template("prediction.html", prediction_text="News  is -> {}".format(predict))


    else:
        return render_template("prediction.html")
    
    
@app.route('/predictionemail', methods=['POST'])
def predictionemail():
    
    if request.method == "POST":
        email = str(request.form['email'])
     
        transformed_sms = transform_text(email)

        vector_input = tfidf.transform([transformed_sms])

        result = email_model.predict(vector_input)[0]

        prediction= "SPAM"
        if result == 0:
            prediction="NOT SPAM"

        return render_template("prediction.html", prediction_text="Email/SMS is -> {}".format(prediction))



if __name__ == '__main__':
    app.debug = True
    app.run()
