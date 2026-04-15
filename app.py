from flask import Flask, request, jsonify
from utils.validators import validate_date_range
from utils.predictions import make_predictions
from utils.model_loader import load_models
import traceback

app = Flask(__name__)

# Load both Prophet dan LightGBM models
try:
    models = load_models()
    print("✓ Models berhasil dimuat: Prophet + LightGBM")
except Exception as e:
    print(f"✗ Error loading models: {e}")
    traceback.print_exc()
    models = None

@app.route('/', methods=['POST'])
def predict_donation():
    if models is None:
        return jsonify({"status": "error", "message": "Models tidak berhasil dimuat"}), 500
    
    data = request.get_json()
    start_str = data.get('start_date')
    end_str = data.get('end_date')

    if not start_str or not end_str:
        return jsonify({"status": "error", "message": "Parameter start_date dan end_date wajib diisi."}), 400

    is_valid, start_date, end_date, error_msg = validate_date_range(start_str, end_str)
    
    if not is_valid:
        return jsonify({"status": "error", "message": error_msg}), 400

    try:
        result = make_predictions(models, start_date, end_date)
        
        return jsonify({
            "status": "success",
            "data": result
        }), 200

    except Exception as e:
        error_msg = str(e)
        print(f"✗ Error in prediction: {error_msg}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": error_msg}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    if models is None:
        return jsonify({"status": "unhealthy", "message": "Models tidak dimuat"}), 503
    
    return jsonify({
        "status": "healthy",
        "models": ["Prophet", "LightGBM"],
        "ensemble_method": "Weighted Average (70% Prophet + 30% LightGBM)"
    }), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)