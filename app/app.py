from flask import Flask, render_template, request, redirect, url_for
from app.model_utils import load_model, load_vectorizer


app = Flask(__name__, template_folder='templates', static_folder='static')

model = load_model()
vectorizer = load_vectorizer()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text', '')
    if not text.strip():
        return redirect(url_for('home'))

    X = vectorizer.transform([text])
    pred = model.predict(X)[0]

    # score (PassiveAggressiveClassifier has decision_function)
    try:
        score = model.decision_function(X)[0]
    except:
        score = None

    return render_template('result.html', text=text, prediction=pred, score=score)

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True, host='127.0.0.1', port=5000)
