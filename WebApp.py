
from flask import Flask, render_template, request
import pickle

with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

app = Flask(__name__)

@app.route("/", methods = ["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        comment = request.form["comment"]
        if comment:
            vec = vectorizer.transform([comment])
            result = model.predict(vec)[0]
            prediction = result.capitalize()
    return render_template("main.html", prediction = prediction)
if __name__ == "__main__":
    app.run(debug=True)