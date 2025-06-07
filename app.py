import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

app = Flask(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_ID = os.getenv("GROQ_MODEL_ID", "llama-3.3-70b-versatile") # Default Groq model

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")

client = Groq(api_key=GROQ_API_KEY)

# --- Assistant Persona and LLM Interaction ---
SYSTEM_PERSONA = """
You are a Coder, named as Ego, an expert AI assistant.
You are a master of frontend development (HTML, CSS, JavaScript, React, Angular, Vue, etc.),
AI/ML (Python, TensorFlow, PyTorch, scikit-learn, NLP, Computer Vision, etc.),
and all computer programming languages (Python, Java, C++, C#, Go, Rust, Ruby, PHP, Swift, Kotlin, SQL, etc.).

Your capabilities include:
- Generating code in any specified language for a given task.
- Explaining code snippets clearly and concisely.
- Helping edit and refactor existing code, providing suggestions and improvements.
- Teaching programming languages and concepts with examples.
- Answering general questions related to computer science, software development, frontend, and AI/ML.

The user will provide their request. If they include a code snippet, it will be clearly marked.
Focus on fulfilling the user's request directly.
When generating code, provide it in a clear, well-formatted way, often using markdown code blocks.
When explaining, be thorough but easy to understand.
When teaching, break down complex topics into simpler parts.
Be helpful, patient, and accurate.
"""

def ask_llm_groq(user_full_prompt, max_new_tokens=1500, temperature=0.7):
    """
    Sends a request to the Groq API and returns its response.
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PERSONA,
                },
                {
                    "role": "user",
                    "content": user_full_prompt,
                },
            ],
            model=GROQ_MODEL_ID,
            temperature=temperature,
            max_tokens=max_new_tokens,
            top_p=1,
            stop=None,
            stream=False,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Groq API Error: {e}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_polyglot():
    data = request.get_json()
    user_query = data.get('user_query', '')
    code_snippet = data.get('code_snippet', '')

    if not user_query:
        return jsonify({"error": "User query cannot be empty."}), 400

    # Construct the full prompt for the LLM
    full_user_prompt = user_query
    if code_snippet and code_snippet.strip():
        # Clearly demarcate the code snippet for the LLM
        full_user_prompt += f"\n\n--- Code Snippet Provided ---\n```\n{code_snippet}\n```\n--- End Code Snippet ---"

    # Get the response from the LLM
    print(f"Sending to LLM ({GROQ_MODEL_ID}): {full_user_prompt[:200]}...") # Log a snippet
    ai_response = ask_llm_groq(full_user_prompt)
    print(f"LLM Response: {ai_response[:200]}...") # Log a snippet

    return jsonify({"response": ai_response})

if __name__ == '__main__':
    app.run(debug=True, port=5001)