from flask import Flask, request, jsonify, render_template
from transformers import pipeline

app = Flask(__name__)
chatbot = pipeline("text-generation", model="distilgpt2")

@app.route("/")
def index():
    return render_template("index.html")  # Serves the chatbot UI

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_message = data.get("message", "").strip()
        if not user_message:
            return jsonify({"response": "Please enter a message."})
        reply = chatbot(user_message, max_length=100, num_return_sequences=1)[0]["generated_text"]
        return jsonify({"response": reply})
    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
