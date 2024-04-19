from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the models
classifier = pickle.load(open('Movie-Genre-Prediction-main/model.pkl', 'rb'))
cv = pickle.load(open('Movie-Genre-Prediction-main/cv.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vector = cv.transform(data).toarray()
        my_prediction = classifier.predict(vector)
        return render_template('index.html', prediction=my_prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
