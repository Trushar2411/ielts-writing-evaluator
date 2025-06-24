# BERT - Bidirectional Encoder Representations from Transformers

BERT is a transformer-based language model developed by Google that excels at understanding the context of natural language by analyzing text bidirectionally—considering both left and right contexts simultaneously.

This project focuses on using a BERT-based model to automatically predict scores for IELTS Writing Task 2 essays and provide targeted writing feedback. It includes both regression and classification approaches. In the regression setting, the model predicts a continuous score such as 6.5 or 7.0, closely mimicking the IELTS band scale. In classification, the model categorizes essays into discrete band levels like "less than 5", "5", "6", "7", or "8 and above". This dual approach enables flexibility depending on the application—whether precise scoring is required or simple band categorization is sufficient.

The model is built on top of the `bert-base-uncased` transformer architecture and is fine-tuned using a dataset of essays and their associated scores. For the regression task, the model uses mean squared error as the loss function, while for classification, it uses cross-entropy loss with multiple band-level labels. The dataset is tokenized using the BERT tokenizer, and the model is trained using the Hugging Face `Trainer` API for simplicity and extensibility.

To help visualize model performance, the project includes several analysis tools. For regression, a scatter plot shows predicted vs. actual scores, including a fitted regression line and a reference line for ideal predictions. For classification, a confusion matrix heatmap provides insight into how often essays are correctly or incorrectly classified into different band levels.

A unique feature in the regression workflow is the ability to automatically identify essays with the highest prediction errors. These are the samples where the difference between the predicted score and the actual score is the largest. This is useful for model analysis, error diagnosis, and identifying edge cases where the model struggles—helping developers understand limitations and possibly guiding future data augmentation or model improvement strategies.

An additional feature of this project is a feedback module. Based on the predicted band score, the model provides writing suggestions to help users understand how to improve their essays. For example, lower bands might receive suggestions to work on grammar and clarity, while higher bands are encouraged to refine lexical resources and maintain argument structure. This makes the tool not just evaluative but also instructive, supporting learners in their development.

The model also supports saving and loading, allowing it to be reused for inference or deployed in an application. This is useful for educational platforms or researchers interested in further fine-tuning or analysis.

Future enhancements may include deploying the model via a web interface, and expanding support for question generation using models like T5.

This project demonstrates how modern NLP techniques can be used in real-world educational contexts to automate assessment and guide student improvement in a personalized and scalable manner.

## Model Performance

Regression Results

The regression model showed consistent improvement over training steps, with training loss decreasing from 28.39 at step 10 to 0.44 at step 240. This significant drop indicates that the model learned to approximate the score distribution effectively. Final performance on the test set was evaluated using:

    Mean Squared Error (MSE): 0.643

    Mean Absolute Error (MAE): 0.650

These values suggest that the model, on average, predicted scores within approximately ±0.65 of the true band. For a high-stakes context like IELTS, this level of error is reasonably low and provides a reliable continuous scoring mechanism.

Classification Results

The classification model aimed to simplify the output by grouping essays into predefined band categories. Training loss reduced from 1.52 to 0.68, with an average training loss of 1.02 by the final epoch. The loss curve showed stable convergence with occasional fluctuations, suggesting general learning with some class-specific variance.

Qualitative inspection of predictions shows the model performs well in assigning essays to appropriate band levels, particularly distinguishing lower bands. Integration with a feedback generator further enhances interpretability by providing writing tips along with predicted bands.

## Notebook Visibility Note

⚠️ **Important**: The Jupyter notebook in this repository may not be visible on GitHub due to the following error:  
`There was an error rendering your Notebook: the 'state' key is missing from 'metadata.widgets'. Add 'state' to each, or remove 'metadata.widgets'.`

However, you can still:
1. Download the notebook file directly
2. Use it on Google Colab (recommended for full functionality)

## Usage Instructions

1. **Google Colab Requirement**: This notebook is designed to run on Google Colab due to Hugging Face Trainer dependencies
2. **Manual File Setup**: Before running the notebook each time:
   - You need to manually upload the `filter_dataset.csv` file
   - Place it in the appropriate directory as referenced in the notebook
