from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import pandas as pd
import json

df = pd.read_csv("data/cleaned_SuicideDetection.csv")

train_df, val_df = train_test_split(df, test_size=0.3)
train_df = train_df.dropna(subset=['cleaned_text'])
val_df = val_df.dropna(subset=['cleaned_text'])
train_df.to_csv("data/train_data.csv", index=False)
val_df.to_csv("data/val_data.csv", index=False)


tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_df['cleaned_text'])
tokenizer_json = tokenizer.to_json()
with open("embeddings/tokenizer.json", "w") as f:
    json.dump(tokenizer_json, f)
print("Completed")

