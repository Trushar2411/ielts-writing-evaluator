from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = FastAPI()

# Load model and tokenizer
model_name_or_path = "./ielts_model"  # Path to your fine-tuned model
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token
model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
model.eval()  # Set model to evaluation mode

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.get("/", response_class=HTMLResponse)
def home():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>IELTS Question Generator</title>
      <style>
        body {
          font-family: Arial, sans-serif;
          max-width: 700px;
          margin: 40px auto;
          padding: 0 20px;
          line-height: 1.5;
          background-color: #f9f9f9;
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
          background: #fff;
          padding: 15px;
          border-radius: 5px;
        }
        button {
          background-color: #2980b9;
          color: white;
          border: none;
          padding: 10px 20px;
          font-size: 1rem;
          cursor: pointer;
          border-radius: 5px;
          display: block;
          margin: auto;
        }
        button:hover {
          background-color: #1c5980;
        }
      </style>
    </head>
    <body>
      <h1>IELTS Question Generator</h1>
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
        // Fetch a question when the page loads
        fetchQuestion();
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

@app.get("/generate")
def generate_question():
    prompt = "IELTS question:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=80,
            do_sample=True,
            temperature=0.9,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    question = generated_text[len(prompt):].strip()

    return {"question": question}
