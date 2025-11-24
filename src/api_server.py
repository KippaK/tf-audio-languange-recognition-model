import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings('ignore')

from flask import Flask, request, jsonify
from flask_cors import CORS
import soundfile as sf
import numpy as np
import tempfile
from main import LanguageDetector

app = Flask(__name__)
CORS(app)  # allow all origins during local dev

# Allow up to 50 MB uploads
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# Load the AI model
try:
    detector = LanguageDetector("best_model_tf")
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    detector = None

@app.route("/health", methods=["GET"])
def health():
    """Check backend and model status"""
    return jsonify({"status": "ok", "model_loaded": detector is not None})

@app.route("/analyze", methods=["POST"])
def analyze_audio():
    """Analyze an uploaded WAV file using the AI model"""
    if detector is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        if not file.filename.lower().endswith(".wav"):
            return jsonify({"error": "Only WAV files are supported"}), 400

        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        try:
            # Load and preprocess
            audio, sample_rate = sf.read(tmp_path)
            processed_audio = detector.preprocess_audio(audio)
            mel_spec = detector.audio_to_mel(processed_audio)

            # Predict
            predictions = detector.model.predict(mel_spec, verbose=0)
            confidence = float(np.max(predictions[0]))
            class_idx = int(np.argmax(predictions[0]))
            detected_language = detector.labels[class_idx]

            # Build result dictionary
            confidences = {
                detector.labels[i]: float(predictions[0][i])
                for i in range(len(detector.labels))
            }

            return jsonify({
                "success": True,
                "detected_language": detected_language,
                "confidence": confidence,
                "all_confidences": confidences,
                "audio_duration": float(len(audio) / sample_rate),
                "sample_rate": int(sample_rate)
            })

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Simple HTTP server — no SSL to avoid macOS issues
    app.run(debug=True, host="0.0.0.0", port=5001)
