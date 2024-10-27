from torch.utils.data import Dataset
from tensorflow.keras.preprocessing.sequence import pad_sequences
class SuicideDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, maxlen= 100):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        label = self.labels.iloc[idx]
        sequence = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=self.maxlen, padding="post")
        return padded[0], label
