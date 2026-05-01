import torch
from torchvision import transforms
from PIL import Image
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import get_model

# ─── Setup ────────────────────────────────────────────────────────────────────
device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'detector.pth')

# Must match the validation transforms used in dataset.py
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Class labels — ImageFolder sorts alphabetically: fake=0, real=1
CLASSES = ['fake', 'real']


def load_model():
    """Load the trained model from disk."""
    model = get_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    model.to(device)
    return model


def predict(image_input, model):
    """
    Predict whether an image is AI-generated or real.
    
    Args:
        image_input: A file path (str) or a PIL Image object
        model: Loaded model from load_model()
    
    Returns:
        dict with keys:
            - label: 'AI Generated' or 'Real'
            - confidence: float 0-100 (e.g. 94.7)
            - fake_prob: probability of being fake (0-1)
            - real_prob: probability of being real (0-1)
    """
    # Accept both file path and PIL Image
    if isinstance(image_input, str):
        image = Image.open(image_input).convert('RGB')
    else:
        image = image_input.convert('RGB')

    tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    fake_prob = float(probs[0])
    real_prob = float(probs[1])

    # Determine label: fake=0, real=1
    if fake_prob > real_prob:
        label      = 'AI Generated'
        confidence = fake_prob * 100
    else:
        label      = 'Real'
        confidence = real_prob * 100

    return {
        'label':      label,
        'confidence': round(confidence, 1),
        'fake_prob':  round(fake_prob * 100, 1),
        'real_prob':  round(real_prob * 100, 1),
    }


# ─── Quick test if run directly ───────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    print("Loading model...")
    m = load_model()
    result = predict(sys.argv[1], m)
    print(f"\nResult: {result['label']}")
    print(f"Confidence: {result['confidence']}%")
    print(f"  Fake probability: {result['fake_prob']}%")
    print(f"  Real probability: {result['real_prob']}%")