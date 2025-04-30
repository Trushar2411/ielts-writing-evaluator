# 📝 IELTS Writing Task 2 - Dataset Collection & Evaluation Model

## 📚 Overview

This project focuses on building a robust NLP pipeline for **IELTS Writing Task 2**. Our primary objectives are:

1. **Dataset Collection & Cleaning**: Filtering and preprocessing a publicly available dataset of IELTS essays.
2. **Question Generation Model**: Training a language model that can generate IELTS Writing Task 2-style questions.
3. **Essay Evaluation Model**: Creating a model that can evaluate and grade user-written essays based on trained data.

> ⚠️ *Note: This project currently focuses only on Writing Task 2. Task 1 will be added in future iterations.*

---

## 👨‍💻 Project Members

- **Trushar Ghanekar**
- **Hoan Vu**
- **Prachi Sheth**

> We are students of the **HBRS** program, collaborating to apply NLP techniques to real-world educational challenges.

---

## 📂 Dataset

We are currently using the following Kaggle dataset for training and testing:

🔗 [IELTS Writing Scored Essays Dataset - by Mazlumi](https://www.kaggle.com/datasets/mazlumi/ielts-writing-scored-essays-dataset/data)

This dataset includes:
- Writing Task 2 questions
- Human-written responses
- Band scores across multiple categories

---

## ✅ Project Phases

### 1. Data Preprocessing (🛠️ Current Phase)
- Load and inspect the dataset
- Filter out incomplete or low-quality entries
- Clean essay text (remove special characters, normalize casing, etc.)
- Prepare structured input for downstream modeling

### 2. Question Generation (🧠 Upcoming)
- Use NLP/Transformer models (e.g. GPT) to generate new IELTS-style Task 2 questions
- Train on real prompts extracted from the dataset

### 3. Essay Grading (📊 Future Work)
- Build a grading model to evaluate user-generated essays
- Predict band scores based on trained scoring metrics

---

## 🛠️ Technologies & Tools

- Python
- Pandas, NumPy (Data processing)
- NLTK / spaCy (NLP Preprocessing)
- Hugging Face Transformers (for future modeling)
- Scikit-learn / PyTorch / TensorFlow (for evaluation model)
- Kaggle API (for data access)

---

## 🚀 Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-repo/ielts-writing-evaluator.git
   cd ielts-writing-evaluator
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download dataset manually from Kaggle**  
   - Upload the downloaded dataset to the `data/` directory
---

## 📌 Future Plans

- Expand dataset with Task 1 content
- Add support for feedback generation along with grading
- Deploy a web-based platform for users to practice IELTS writing with instant AI evaluation

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
