# Importing essential libraries
from flask import Flask, render_template, request
import pickle

# Load the Multinomial Naive Bayes model and CountVectorizer object from disk
filename = 'email_spam_model.pkl' # stream of bytes
cv_filename = 'cv.pkl'

classifier = pickle.load(open(filename,'rb')) # unpickling - Object creation
cv=pickle.load(open(cv_filename,'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = classifier.predict(vect)
        return render_template('result.html', prediction=my_prediction)
    
if __name__ == '__main__':
    app.run()