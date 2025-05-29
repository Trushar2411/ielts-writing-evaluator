# BERT - Bidirectional Encoder Representations from Transformers

This project uses the BERT model to predict scores for IELTS Writing Task 2 essays. 

BERT is a transformer-based language model developed by Google that excels at understanding the context of natural language by analyzing text bidirectionally—considering both left and right contexts simultaneously.

In this work, the pretrained `bert-base-uncased` model is fine-tuned on a dataset of IELTS essays and their corresponding overall scores using a regression approach. The model is trained to map each essay to a continuous score, enabling automated scoring that approximates human evaluation. The training process involves tokenizing the essays, constructing a custom PyTorch dataset, and optimizing the model using Hugging Face's `Trainer` API. After training, the model's performance is evaluated using standard metrics such as Mean Squared Error (MSE) and Mean Absolute Error (MAE), providing insight into prediction accuracy.

