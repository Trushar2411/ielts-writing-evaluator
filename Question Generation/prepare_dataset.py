import pandas as pd

# Read the CSV assuming no header
df = pd.read_csv("cleaned_ielts_questions.csv", header=None)

# Save it as a text file with one question per line
df[0].dropna().to_csv("cleaned_ielts_questions.txt", index=False, header=False)

print("✅ Text file saved as 'cleaned_ielts_questions.txt'")