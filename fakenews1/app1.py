from flask import Flask, request, render_template
from joblib import load
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

clf = load('model.joblib')
vectorizer = load('vectorizer.joblib')
def preprocess_title(title):
    # Lowercase the title
    title = title.lower()

    # Remove punctuation and digits
    title = title.translate(str.maketrans('', '', string.punctuation + string.digits))

    # Tokenize the title
    words = word_tokenize(title)

    # Remove stop words
    words = [word for word in words if word not in stop_words]

    # Stem or lemmatize the words
    words = [stemmer.stem(word) for word in words]
   
        # Join the words back into a string
    title = ' '.join(words)

    return title


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    title = request.form['title']
    preprocessed_title = preprocess_title(title)
    X = vectorizer.transform([preprocessed_title])
    y_pred = clf.predict(X)
    if y_pred[0]== 1:
        result = 'real'
    else:
        result = 'fake'
    return render_template('result.html', result=result, title=title)

if __name__ == '__main__':
    app.run(debug=True)