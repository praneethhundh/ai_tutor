from transformers import pipeline, set_seed

# Load the GPT-2 model (free)
chatbot = pipeline("text-generation", model="gpt2")
set_seed(42)  # optional: for reproducibility

# Mimic OpenAI's message format
def get_ai_response(user_message):
    system_prompt = "You are a friendly and interactive English tutor for kids.\n"
    full_prompt = system_prompt + "User: " + user_message + "\nTutor:"
    
    response = chatbot(full_prompt, max_length=150, num_return_sequences=1, pad_token_id=50256)
    generated_text = response[0]["generated_text"]
    
    # Extract only tutor response
    tutor_reply = generated_text.split("Tutor:")[-1].strip().split("User:")[0].strip()
    return tutor_reply
