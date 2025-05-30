## 📁 Project Workflow Overview

### 1. Dataset Preparation

- The dataset (`cleaned_ielts_questions.csv`) contains IELTS writing questions.
- Duplicates were removed.
- The CSV has no headers and one question per line.
- Converted to plain text format (`cleaned_ielts_questions.txt`) for fine-tuning.

---

### 2. Fine-Tuning the GPT-2 Model

- Used Hugging Face's GPT-2 model and tokenizer.
- Trained on the cleaned dataset for a few epochs.
- The fine-tuned model and tokenizer were saved in the `ielts_model/` directory.

---

### 🌐 3. Run the FastAPI Server

#### ✅ Step 1: Install FastAPI and Uvicorn

```bash
pip install fastapi uvicorn
```

#### ✅ Step 2: Run the API

Ensure `api.py` is in the same folder as `ielts_model/`, then start the server:

```bash
python -m uvicorn api:app --reload
```

---

### 🖥 4. Open the Web Interface

Go to your browser and open:

```
http://127.0.0.1:8000/
```

You’ll see:

* A heading: **IELTS Test**
* A randomly generated IELTS question
* A **Generate New Question** button to refresh and get a new question

---

### 🔁 Example Output

```json
{
  "question": "Some people believe the purpose of education is to prepare individuals to be useful members of society. Discuss both views and give your opinion."
}
```

---

## ✅ Requirements

Install the following Python packages:

```text
torch
transformers
datasets
fastapi
uvicorn
```

You can create a `requirements.txt` file with the above and install using:

```bash
pip install -r requirements.txt
```

---

## 🚀 Future Improvements

* Deploy the app online (e.g., Render, Hugging Face Spaces)
* Add input prompts for topic-controlled generation
* Collect user feedback or allow saving questions
* Add logging or analytics to track usage

---

## 📌 Project Structure

```
Question Generation/
├── cleaned_ielts_questions.csv
├── cleaned_ielts_questions.txt
├── fine_tune.py
├── api.py
├── ielts_model/
└── index.html
```


```
