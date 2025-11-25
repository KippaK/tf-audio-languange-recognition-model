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
    print("Loading model from best_model_tf...")
    # Model is in parent directory (one level up from src/)
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "best_model_tf")
    detector = LanguageDetector(model_path)
    print("✓ Model loaded successfully")
    print(f"  - Sample rate: {detector.sample_rate}")
    print(f"  - Languages: {detector.labels}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    import traceback
    traceback.print_exc()
    detector = None

@app.route("/health", methods=["GET"])
def health():
    """Check backend and model status"""
    return jsonify({"status": "ok", "model_loaded": detector is not None})

@app.route("/analyze", methods=["POST"])
def analyze_audio():
    """Analyze an uploaded WAV file using the AI model"""
    if detector is None:
        return jsonify({"success": False, "error": "Model not loaded"}), 500

    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"success": False, "error": "No file selected"}), 400
        if not file.filename.lower().endswith(".wav"):
            return jsonify({"success": False, "error": "Only WAV files are supported"}), 400

        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        try:
            # Load and preprocess
            audio, sample_rate = sf.read(tmp_path)

            # Handle sample rate conversion if needed
            if sample_rate != detector.sample_rate:
                if sample_rate > detector.sample_rate:
                    step = sample_rate // detector.sample_rate
                    audio = audio[::step]
                    sample_rate = detector.sample_rate

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

            # Determine true language from filename if it follows naming convention
            from config import PREFIX_MAP
            true_language = None
            is_correct = None

            for prefix, lang in PREFIX_MAP.items():
                if file.filename.startswith(prefix):
                    true_language = lang
                    is_correct = (detected_language == true_language)
                    break

            return jsonify({
                "success": True,
                "filename": file.filename,
                "detected_language": detected_language,
                "confidence": confidence,
                "all_confidences": confidences,
                "audio_duration": float(len(audio) / sample_rate),
                "sample_rate": int(sample_rate),
                "true_language": true_language,
                "is_correct": is_correct
            })

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except Exception as e:
        print(f"Error in analyze_audio: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/test-batch", methods=["GET"])
def test_batch():
    """Run the same batch test as main.py - tests all files in data/test/"""
    if detector is None:
        return jsonify({"error": "Model not loaded"}), 500

    from config import PREFIX_MAP

    # Use absolute path - data/test is in parent directory
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, "data", "test")
    transcript_file = os.path.join(data_dir, "transcripts.txt")

    if not os.path.exists(data_dir):
        return jsonify({"error": f"Test directory not found: {data_dir}"}), 404

    # Load transcripts
    transcripts = {}
    try:
        with open(transcript_file, 'r', encoding='utf-8') as f:
            for line in f:
                if '\t' in line:
                    filename, transcript = line.strip().split('\t', 1)
                    transcripts[filename] = transcript
    except Exception as e:
        print(f"Error loading transcripts: {e}")

    # Find WAV files
    wav_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.wav')]

    if not wav_files:
        return jsonify({"error": "No WAV files found"}), 404

    print(f"Found {len(wav_files)} WAV files to process")

    results = []
    correct_predictions = 0

    # Process all test files
    for filename in sorted(wav_files):
        filepath = os.path.join(data_dir, filename)
        language, confidence = detector.predict(filepath)

        if language:
            # Determine true language from filename
            true_language = None
            for prefix, lang in PREFIX_MAP.items():
                if filename.startswith(prefix):
                    true_language = lang
                    break

            # Get transcript
            transcript = transcripts.get(filename, "No transcript available")

            # Check correctness
            is_correct = (language == true_language) if true_language else None
            if is_correct:
                correct_predictions += 1

            results.append({
                "filename": filename,
                "predicted_language": language,
                "confidence": confidence,
                "true_language": true_language,
                "is_correct": is_correct,
                "transcript": transcript
            })

    # Calculate accuracy
    accuracy = (correct_predictions / len(wav_files) * 100) if wav_files else 0

    print(f"Batch test complete: {correct_predictions}/{len(wav_files)} correct ({accuracy:.1f}%)")

    return jsonify({
        "success": True,
        "total_files": len(wav_files),
        "correct_predictions": correct_predictions,
        "accuracy": accuracy,
        "results": results
    })


if __name__ == "__main__":
    # Simple HTTP server — no SSL to avoid macOS issues
    app.run(debug=True, host="0.0.0.0", port=5001)
