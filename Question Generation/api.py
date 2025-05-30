from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = FastAPI()

model_name_or_path = "./ielts_model"  # your fine-tuned model folder

tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
model = GPT2LMHeadModel.from_pretrained(model_name_or_path)

tokenizer.pad_token = tokenizer.eos_token  # for GPT-2

@app.get("/", response_class=HTMLResponse)
def home():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>IELTS Test</title>
      <style>
        body {
          font-family: Arial, sans-serif;
          max-width: 700px;
          margin: 40px auto;
          padding: 0 20px;
          line-height: 1.5;
        }
        h1 {
          text-align: center;
          color: #2c3e50;
          margin-bottom: 30px;
        }
        #question {
          font-size: 1.2rem;
          color: #34495e;
          border-left: 4px solid #2980b9;
          padding-left: 15px;
          margin-bottom: 20px;
        }
        button {
          background-color: #2980b9;
          color: white;
          border: none;
          padding: 10px 20px;
          font-size: 1rem;
          cursor: pointer;
          border-radius: 5px;
        }
        button:hover {
          background-color: #1c5980;
        }
      </style>
    </head>
    <body>
      <h1>IELTS Test</h1>
      <div id="question">Loading question...</div>
      <button onclick="fetchQuestion()">Generate New Question</button>

      <script>
        async function fetchQuestion() {
          const qDiv = document.getElementById('question');
          qDiv.textContent = 'Loading question...';
          try {
            const response = await fetch('/generate');
            const data = await response.json();
            qDiv.textContent = data.question || 'No question generated.';
          } catch (error) {
            qDiv.textContent = 'Failed to load question.';
            console.error('Error fetching question:', error);
          }
        }
        // Fetch question on page load
        fetchQuestion();
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

@app.get("/generate")
def generate_question():
    prompt = "IELTS question:"
    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        pad_token_id=tokenizer.eos_token_id,
        max_length=60,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        temperature=0.8,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    question = generated_text[len(prompt):].strip()

    return {"question": question}
