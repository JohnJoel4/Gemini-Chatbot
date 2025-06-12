import gradio as gr
import google.generativeai as genai
import os

# ==========================
# App Configuration
# ==========================
APP_TITLE = "✨ Gemini Chatbot"
APP_DESCRIPTION = (
    "Talk with a conversational AI powered by Google's **Gemini-1.5-Flash** model. "
    "This chatbot maintains conversation history for more natural interactions."
)
# ==========================
# Model and API Setup
# ==========================
try:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("❌ GOOGLE_API_KEY secret not found!")
    
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction="You are a friendly and helpful chatbot.",
        safety_settings={
            'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
            'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
            'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
            'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'
        }
    )

    chat = model.start_chat(history=[])
    MODEL_LOADED = True
    print("✅ Model loaded and chat started successfully.")

except Exception as e:
    print(f"🔴 Error loading model: {e}")
    MODEL_LOADED = False

# ==========================
# Chatbot Response Function
# ==========================
def respond(message, history):
    if not MODEL_LOADED:
        yield "🔴 Error: The AI model could not be loaded. Please check the API key."

    try:
        response = chat.send_message(message, stream=True)
        full_response = ""
        for chunk in response:
            full_response += chunk.text
            yield full_response
    except Exception as e:
        yield f"❌ An error occurred: {e}"

# ==========================
# Gradio UI Setup
# ==========================
demo = gr.ChatInterface(
    fn=respond,
    chatbot=gr.Chatbot(
        height=500,
        label="💬 Gemini Chatbot"
    ),
    textbox=gr.Textbox(
        placeholder="Type your message here...",
        container=False,
        scale=7
    ),
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    theme="soft",
    examples=[
        "What are some fun things to do in St. Louis, Missouri?",
        "Write a short, funny poem about a robot who loves to cook.",
        "Explain the concept of zero-shot learning in simple terms.",
    ],
    cache_examples=False,
)

# ==========================
# Run the App
# ==========================
if __name__ == "__main__":
    demo.launch()
