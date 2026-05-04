import os
import uuid
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import sys

# Allow imports from src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.predict import load_model, predict

# ─── Flask App Setup ──────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = 'ai-detector-secret-key-2024'  # Needed for Flask sessions

UPLOAD_FOLDER    = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_EXTS     = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
MAX_FILE_MB      = 16

app.config['UPLOAD_FOLDER']    = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_MB * 1024 * 1024
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ─── Load Model Once at Startup ───────────────────────────────────────────────
print("Loading AI detection model...")
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'detector.pth')

detector_model = None
model_loaded   = False

if os.path.exists(MODEL_PATH):
    try:
        detector_model = load_model()
        model_loaded   = True
        print("[OK] Detection model loaded successfully!")
    except Exception as e:
        print(f"[X] Could not load model: {e}")
        print("  -> Run 'python src/train.py' first to train the model.")
else:
    print("[X] No trained model found at models/detector.pth")
    print("  -> Run 'python src/train.py' first to train the model.")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTS


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the main chat UI."""
    return render_template('index.html')


@app.route('/api/status')
def status():
    """Check if model and Ollama are ready."""
    return jsonify({
        'model_loaded': model_loaded,
        'model_path':   MODEL_PATH,
    })


@app.route('/api/upload', methods=['POST'])
def upload_image():
    """
    Upload an image, run the detector, return the result.
    The image is saved locally so Ollama can read it.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not supported. Use: {", ".join(ALLOWED_EXTS)}'}), 400

    if not model_loaded:
        return jsonify({'error': 'Model not loaded. Run training first: python src/train.py'}), 503

    # Save the file
    filename    = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
    image_path  = os.path.join(UPLOAD_FOLDER, filename)
    file.save(image_path)

    # Run AI detection
    try:
        pil_image       = Image.open(image_path).convert('RGB')
        detection_result = predict(pil_image, detector_model)
    except Exception as e:
        return jsonify({'error': f'Detection failed: {str(e)}'}), 500

    # (Chat is disabled in this deployment — detector only)

    # Return result + a URL to display the image in the UI
    return jsonify({
        'success':    True,
        'result':     detection_result,
        'image_url':  f'/uploads/{filename}',
        'image_path': image_path,
    })


@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat is disabled in this deployment (detector-only mode)."""
    return jsonify({
        'reply': '🤖 Chat is disabled in this demo deployment. Upload an image above to get the AI vs Real detection result!'
    })


@app.route('/api/reset', methods=['POST'])
def reset_chat():
    """No-op in detector-only mode."""
    return jsonify({'success': True, 'message': 'Conversation cleared!'})


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded images."""
    return send_from_directory(UPLOAD_FOLDER, filename)


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Railway sets PORT automatically
    print("\n" + "="*50)
    print("  AI Image Detector — Detector Mode")
    print("="*50)
    print(f"  Open in browser: http://localhost:{port}")
    print(f"  Model status:    {'[OK] Ready' if model_loaded else '[X] Not trained yet'}")
    print("="*50 + "\n")
    app.run(debug=False, host='0.0.0.0', port=port)
