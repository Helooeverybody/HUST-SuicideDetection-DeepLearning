import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import json
from tensorflow.keras.preprocessing.text import Tokenizer,tokenizer_from_json
import random
import pandas as pd
from label_encoder_wunk import LabelEncoderWithUnk

# Initialize the constants
batch_size = 64
embedding_dim = 300

def generate_batch(data, batch_size, num_skips, skip_window):
    centers = []
    contexts = []
    for sentence in data:
        sentence_length = len(sentence)
        if sentence_length < (2 * skip_window + 1):
            continue  # Skip sentences that are too short for the given window size

        # Generate (center, context) pairs for the sentence
        for i in range(skip_window, sentence_length - skip_window):
            center = sentence[i]
            context_indices = list(range(i - skip_window, i + skip_window + 1))
            context_indices.remove(i)  # Remove the center word index
            context_words = [sentence[idx] for idx in context_indices]
            
            # Generate samples based on num_skips
            random_contexts = random.sample(context_words, min(num_skips, len(context_words)))
            for context in random_contexts:
                centers.append(center)
                contexts.append(context)

        # Break if have enough samples for the batch
        if len(centers) >= batch_size:
            break

    # If do not have enough samples, pad with zero
    centers.extend([0] * (batch_size - len(centers)))
    contexts.extend([0] * (batch_size - len(contexts)))

    centers = torch.LongTensor(centers[:batch_size])
    contexts = torch.LongTensor(np.array(contexts[:batch_size]).reshape(batch_size, 1))
    return centers, contexts

class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.linear = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, context_word):
        embs = self.embedding(context_word).mean(dim=1)  # Average over context embeddings
        scores = self.linear(embs)
        log_probs = F.log_softmax(scores, dim=1)
        return log_probs

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim,num_neg_samples=3):
        super(SkipGramModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.num_neg_samples = num_neg_samples

    def forward(self, centers, contexts):
        batch_size = len(centers)
        u_embeds = self.embedding(centers).view(batch_size, 1, -1)
        v_embeds = self.embedding(contexts).view(batch_size, -1, 1)
        pos_scores = torch.bmm(u_embeds, v_embeds).squeeze()
        
         # Negative sampling
        neg_samples = self.generate_negative_samples(centers)
        neg_embeds = self.embedding(neg_samples)  #(batch_size, num_neg_samples, embedding_dim)
       
        neg_scores = torch.bmm(u_embeds, neg_embeds.transpose(1, 2)).squeeze()  #  (batch_size, num_neg_samples)

        pos_loss = -F.logsigmoid(pos_scores).mean()
        neg_loss = -F.logsigmoid(-neg_scores).mean()
        
        loss = pos_loss + neg_loss
        return loss
    def generate_negative_samples(self, centers):
        # Create a tensor of negative samples
        neg_samples = []
        for center in centers:
            negative_sample = random.sample(range(self.embedding.weight.size(0)), self.num_neg_samples)
            while center.item() in negative_sample:
                negative_sample = random.sample(range(self.embedding.weight.size(0)), self.num_neg_samples)
            neg_samples.append(negative_sample)

        return torch.tensor(neg_samples, device=centers.device)

def train(data, mode, vocab_size, embedding_dim, batch_size, num_skips, skip_window, num_steps, learning_rate, clip, tokenizer):
    if mode == 'CBOW':
        model = CBOWModel(vocab_size, embedding_dim)
    elif mode == 'SkipGram':
        model = SkipGramModel(vocab_size, embedding_dim)
    else:
        raise ValueError("Model not supported!")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for step in range(num_steps):
        centers, contexts = generate_batch(data, batch_size, num_skips, skip_window)
        optimizer.zero_grad()
        if mode == 'CBOW':
            y_pred = model(contexts)
            loss = F.nll_loss(y_pred, centers)
        elif mode == 'SkipGram':
            loss = model(centers, contexts)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        if step %100 == 0:
            print(f"Step {step}, Loss: {loss.item()}")
    model_embeddings = model.embedding.weight.data
    with open(f"embeddings/{mode}_embeddings.pkl", "wb") as f:
        pickle.dump(model_embeddings, f)
    print(f"{mode} embeddings saved !.")
    return model.embedding.weight.data
def build_vocab(texts):
    encoder = LabelEncoderWithUnk(unk_token='<UNK>')
    all_tokens = []
    for text in texts:
        tokens = text.split()
        tokens = [token for token in tokens if len(token) < 100]
        all_tokens.extend(tokens)
    encoder.fit(sorted(set(all_tokens)))
    vocab_size = len(encoder.classes_)
    return vocab_size, encoder
def preprocess_text_data(filename,tokenizer):
    """Preprocess CSV file containing text, returning tokenized data."""
    df = pd.read_csv(filename, usecols=['cleaned_text'], chunksize=1000)  # Load data in chunks
    data = []
    print("Tokenizing data in chunks...")
    for i, chunk in enumerate(df):
        print(f"Processing chunk {i+1}")
        # Tokenize each chunk with the pre-trained tokenizer
        tokenized_texts = tokenizer.texts_to_sequences(chunk['cleaned_text'].tolist())
        data.extend(tokenized_texts)  
    vocab_size = len(tokenizer.word_index) + 1  
    return data, vocab_size, tokenizer
if __name__ == "__main__":
    with open("embeddings/tokenizer.json", "r") as f:
        tokenizer_json = json.load(f)
        tokenizer = tokenizer_from_json(tokenizer_json)
    data, vocab_size, encoder = preprocess_text_data("data/train_data.csv",tokenizer)
    CBOW_embeddings = train(data, 'CBOW', vocab_size, embedding_dim, batch_size, num_skips=2, skip_window=1, num_steps=1000, learning_rate=0.01, clip=5.0,tokenizer=tokenizer)
    SkipGram_embeddings = train(data, 'SkipGram', vocab_size, embedding_dim, batch_size, num_skips=2, skip_window=1, num_steps=1000, learning_rate=0.01, clip=5.0,tokenizer=tokenizer)
    
   