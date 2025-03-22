from flask import Flask, request, send_from_directory, jsonify
from flask_cors import CORS
import pickle
from classes.generate_prompt import *

# Create the Flask app
app = Flask(__name__, static_folder='./static', static_url_path='')

# Enable CORS
CORS(app)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open("./models/final-model/student_model.pkl", "rb") as f:
    model = pickle.load(f).to(device)
with open("./models/final-model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Define the routes
@app.route('/')
def index_page():
    return app.send_static_file('index.html')

@app.route('/<path:path>')
def serve_custom_path(path):
    return send_from_directory('./', path)

@app.route('/predict', methods=['GET'])
def predict():
    query = request.args.get('search', '').strip()  
    label, confidence = get_prediction(query, model, tokenizer, device)
    return jsonify({
        'label': label,
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)