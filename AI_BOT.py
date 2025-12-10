from flask import Flask, render_template, request
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)

# ✅ Login (optional, remove if model is already downloaded)
login(token="***********")

# 🔄 Load Gemma model once
def load_model():
    print("🔄 Loading Gemma 2B... please wait...")
    model_name = "google/gemma-2b-it"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="cpu"
    )
    print("✅ Model loaded!")
    return tokenizer, model

tokenizer, model = load_model()

# ✅ Store chat history in memory
chat_history = []

def gemma_reply(prompt):
    system_prompt = (
        "You are a highly experienced financial advisor. "
        "You give clear, practical advice on budgeting, investing, savings, loans, and money management. "
        "Explain in simple terms, avoid disclaimers, and always stay professional.\n\n"
        "User: "
    )

    final_prompt = system_prompt + user_msg + "\nAdvisor:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    reply = tokenizer.decode(output[0], skip_special_tokens=True)
    if "Assistant:" in reply:
        reply = reply.split("Assistant:")[-1].strip()
    return reply


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_msg = request.form["message"]
        bot_msg = gemma_reply(user_msg)

        chat_history.append(("You", user_msg))
        chat_history.append(("Gemma", bot_msg))

    return render_template("Index.html", chat=chat_history)


if __name__ == "__main__":
    app.run(debug=True)
