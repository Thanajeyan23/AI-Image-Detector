import ollama
import base64
import os
from pathlib import Path


OLLAMA_MODEL = 'llama3.2-vision'

# System prompt — sets the personality and context for the AI
SYSTEM_PROMPT = """You are an AI image analysis assistant. You help users understand whether images are AI-generated or real photographs, and you discuss the visual characteristics that might indicate AI generation.

When a user shares an image, you'll be given the result from a specialized detector model. Use this information along with your own visual analysis to provide helpful, conversational responses.

Be conversational, friendly, and informative. Keep responses concise but insightful. When discussing AI-generated images, mention specific visual artifacts or characteristics you notice (like unnatural textures, lighting inconsistencies, or unusual details). 

Don't be overly technical unless the user asks for it. You're chatting with a beginner who is curious about AI image detection."""


def image_to_base64(image_path: str) -> str:
    """Convert an image file to base64 string for Ollama."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def build_initial_prompt(detection_result: dict, user_message: str) -> str:
    """Build the first message prompt with detection context."""
    label      = detection_result['label']
    confidence = detection_result['confidence']
    fake_pct   = detection_result['fake_prob']
    real_pct   = detection_result['real_prob']

    # Create a verdict emoji
    verdict = "🤖 AI Generated" if label == 'AI Generated' else "📷 Real Photo"

    return f"""The user uploaded an image. Our specialized detector analyzed it with the following result:

**Detection Result:** {verdict}
**Confidence:** {confidence}%
**AI Generated probability:** {fake_pct}%
**Real photo probability:** {real_pct}%

Now the user says: "{user_message}"

Please respond to their message while referencing the detection result. Analyze the image visually and share your observations."""


class ChatSession:
    """
    Manages a persistent multi-turn conversation with llama3.2-vision.
    
    A session stores the full conversation history so the model
    remembers what was said earlier in the chat.
    """

    def __init__(self):
        self.history = []           # Full conversation history
        self.current_image_path = None   # Path to the currently analyzed image

    def reset(self):
        """Start a fresh conversation."""
        self.history = []
        self.current_image_path = None

    def analyze_image(self, image_path: str, detection_result: dict,
                      user_message: str = "What can you tell me about this image?") -> str:
        """
        Start or continue a conversation about a new image.
        Sends the image + detection result to llama3.2-vision.
        """
        self.current_image_path = image_path

        # Build the first message with detection context
        prompt = build_initial_prompt(detection_result, user_message)

        # Encode image to base64
        img_b64 = image_to_base64(image_path)

        # Add user message to history (with image)
        self.history.append({
            'role':    'user',
            'content': prompt,
            'images':  [img_b64],
        })

        return self._call_ollama()

    def chat(self, user_message: str) -> str:
        """
        Continue the conversation (follow-up questions, no new image).
        """
        # Add user message to history
        self.history.append({
            'role':    'user',
            'content': user_message,
        })

        return self._call_ollama()

    def _call_ollama(self) -> str:
        """Send the conversation history to Ollama and get a response."""
        try:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=self.history,
                options={
                    'temperature': 0.7,   # Creativity level (0=focused, 1=creative)
                    'num_predict': 512,   # Max response length in tokens
                }
            )
            assistant_reply = response['message']['content']

            # Add the AI's reply to history for context in future messages
            self.history.append({
                'role':    'assistant',
                'content': assistant_reply,
            })

            return assistant_reply

        except Exception as e:
            error_msg = f"Ollama error: {str(e)}. Make sure Ollama is running (`ollama serve`)."
            return error_msg

    @property
    def message_count(self):
        return len(self.history)
