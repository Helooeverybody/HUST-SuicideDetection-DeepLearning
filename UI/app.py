from flask import Flask, request, jsonify,render_template
import torch
from chat import answer  

app = Flask(__name__)

@app.get("/")
def index_get():
    return render_template("index.html")

# API endpoint to handle user input and respond with model output
@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.json['message']
    response = answer(user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
