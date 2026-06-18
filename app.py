from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.predictions import make_predictions
from utils.model_loader import load_models
import traceback

app = Flask(__name__)
CORS(app)

# Load Models saat startup (hanya sekali)
try:
    models = load_models()
    print("✓ Models berhasil dimuat:", list(models.keys()))
except Exception as e:
    print(f"✗ Error loading models: {e}")
    traceback.print_exc()
    models = {}


def process_prediction(type_name, request_obj):
    if type_name not in models:
        return jsonify({"status": "error", "message": f"Model {type_name} tidak dimuat"}), 500

    model_to_use = models[type_name]

    data = request_obj.get_json() or {}
    months_ahead = int(data.get('months_ahead', 1))

    try:
        # Panggil make_predictions - hanya butuh model dan jumlah bulan
        # History lag/residual sudah tersimpan di dalam file .pkl
        result = make_predictions(model_to_use, months_ahead)

        formatted_result = []
        for r in result:
            formatted_result.append({
                'Tanggal':          r['date'],
                'Hari_Besar_Islam': r.get('hijri_events', '-'),
                'Prediksi_Prophet': float(r.get('prophet_prediction', r['predicted_donation'])),
                'Prediksi_Hybrid':  float(r['predicted_donation'])
            })

        return jsonify({
            "status": "success",
            "data": formatted_result
        }), 200

    except Exception as e:
        error_msg = str(e)
        print(f"✗ Error in prediction: {error_msg}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": error_msg}), 500


@app.route('/predict/income', methods=['POST'])
def predict_income():
    return process_prediction('income', request)


@app.route('/predict/expense', methods=['POST'])
def predict_expense():
    return process_prediction('expense', request)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "loaded_models": list(models.keys())
    }), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)