from flask import Flask, request, jsonify
from flask_cors import CORS
from stroke import Stroke
from obesity import Obesity
from diabetes import Diabetes
from hypertension import Hypertension
import os

app = Flask(__name__)
# Enable CORS for all domains in development
CORS(app, resources={r"/*": {"origins": "*"}})

#**************************************************#

@app.route('/')
def home():
    return "Welcome to the Health API!"

#**************************************************#

@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    if request.method == 'POST':
        data = request.get_json()

        if not isinstance(data, dict):
            return jsonify({"error": "Expected JSON object, but received invalid format"}), 400

        result = Diabetes(data)  # Pass dictionary directly
        return jsonify({"Result": result}), 200
    
    return "Diabetes Page Navigation", 200

#**************************************************#

@app.route('/obesity', methods=['GET', 'POST'])
def obesity():
    if request.method == 'POST':
        data = request.get_json()

        if not isinstance(data, dict):
            return jsonify({"error": "Expected JSON object, but received invalid format"}), 400

        result = Obesity(data)  # Pass dictionary directly
        return jsonify({"Result": result}), 200
    
    return "Obesity Page Navigation", 200

#**************************************************#

@app.route('/hypertension', methods=['GET', 'POST'])
def hypertension():
    if request.method == 'POST':
        data = request.get_json()

        if not isinstance(data, dict):
            return jsonify({"error": "Expected JSON object, but received invalid format"}), 400

        result = Hypertension(data)
        return jsonify(result.to_dict()), 200
    
    return "Hypertension Page Navigation", 200

#**************************************************#

@app.route('/stroke', methods=['GET', 'POST'])
def stroke():
    if request.method == 'POST':
        data = request.get_json()

        if not isinstance(data, dict):
            return jsonify({"error": "Expected JSON object, but received invalid format"}), 400

        result = Stroke(data)  # Pass dictionary directly
        return jsonify({"Result": result}), 200
    
    return "Stroke Page Navigation", 200
#**************************************************#

if __name__ == '__main__':
    # Run the app on 0.0.0.0 to make it accessible from other devices
    port = int(os.environ.get('PORT', 10000))  # 10000 is the fallback default
    app.run(host='0.0.0.0', port=port)
