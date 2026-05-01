import os
import uuid
from flask import Flask, request, jsonify, render_template, session
from werkzeug.utils import secure_filename
from PIL import Image
import sys

# Allow imports from src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.predict import load_model, predict
from src.chat import ChatSession

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

# In-memory store for chat sessions (one per browser session)
# In production you'd use a database, but for local use this is fine
chat_sessions: dict[str, ChatSession] = {}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTS


def get_chat_session() -> ChatSession:
    """Get or create a ChatSession for the current browser session."""
    sid = session.get('session_id')
    if sid not in chat_sessions:
        sid = str(uuid.uuid4())
        session['session_id']  = sid
        chat_sessions[sid]     = ChatSession()
    return chat_sessions[sid]


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

    # Store image path in the chat session
    chat = get_chat_session()
    chat.current_image_path = image_path

    # Return result + a URL to display the image in the UI
    return jsonify({
        'success':    True,
        'result':     detection_result,
        'image_url':  f'/uploads/{filename}',
        'image_path': image_path,
    })


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Send a message to the chatbot.
    If an image was just uploaded, analyze it.
    If it's a follow-up, continue the conversation.
    """
    data    = request.get_json()
    message = data.get('message', '').strip()

    if not message:
        return jsonify({'error': 'Empty message'}), 400

    chat_obj = get_chat_session()

    # Check if this is the first message about a new image
    detection_result = data.get('detection_result')  # Sent from frontend
    image_path       = data.get('image_path')

    try:
        if detection_result and image_path and os.path.exists(image_path):
            # New image analysis — reset session context and analyze
            chat_obj.reset()
            reply = chat_obj.analyze_image(image_path, detection_result, message)
        else:
            # Follow-up message — continue existing conversation
            if chat_obj.message_count == 0:
                reply = "Please upload an image first so I can analyze it for you! 📸"
            else:
                reply = chat_obj.chat(message)

        return jsonify({'reply': reply})

    except Exception as e:
        return jsonify({'error': f'Chat error: {str(e)}'}), 500


@app.route('/api/reset', methods=['POST'])
def reset_chat():
    """Clear the current conversation to start fresh."""
    chat_obj = get_chat_session()
    chat_obj.reset()
    return jsonify({'success': True, 'message': 'Conversation cleared!'})


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded images."""
    from flask import send_from_directory
    return send_from_directory(UPLOAD_FOLDER, filename)


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*50)
    print("  AI Image Detector Chatbot")
    print("="*50)
    print(f"  Open in browser: http://localhost:5000")
    print(f"  Model status:    {'[OK] Ready' if model_loaded else '[X] Not trained yet'}")
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
