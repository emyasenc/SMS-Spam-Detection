from flask import Flask, request, render_template
from pipeline.predict_pipeline import predict

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict_route():
    prediction_text = None
    if request.method == 'POST':
        message = request.form['message']
        predictions = predict([message])
        
        predicted_labels = (predictions > 0.5).astype(int).tolist()
        prediction_text = "Spam" if predicted_labels[0] == [1] else "Not Spam"

    return render_template('index.html', prediction=prediction_text)

if __name__ == "__main__":
    app.run(host="0.0.0.0")