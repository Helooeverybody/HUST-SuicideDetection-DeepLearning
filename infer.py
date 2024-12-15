import torch
from transformers import BertForSequenceClassification
from transformers import BertTokenizer, BertConfig
import torch
import math
from transformers import ElectraTokenizerFast, ElectraForSequenceClassification
import json
import pickle
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
from model.attention_bilstm import BiLSTM_Attention
from model.lstm import SentimentLSTM
from model.rnn import SentimentRNN
from model.gru import GRU
from model.cnn import CNN

import warnings

warnings.filterwarnings("ignore")

# Load configuration file
with open("config.json", "r") as config_file:
    config = json.load(config_file)
    lstm_config = config["LSTM"]
    bilstm_config = config["BiLSTM_Attention"]
    rnn_config = config["RNN"]
    gru_config = config["GRU"]
    cnn_config = config["CNN"]

# Load tokenizer object
with open("embeddings/tokenizer.json", "r") as f:
    tokenizer_json = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_json)
with open("embeddings/CBOW_embeddings.pkl", "rb") as cbow:
    cbow_embeddings = pickle.load(cbow)
with open("embeddings/SkipGram_embeddings.pkl", "rb") as sg:
    sg_embeddings = pickle.load(sg)

VOCAB_SIZE = len(tokenizer.index_word) + 1
# Load the pre-trained Word2Vec model (e.g., Google News vectors)
w2v_model = Word2Vec(
    sentences=common_texts, vector_size=300, window=5, min_count=1, workers=4
)
embedding_dim = w2v_model.vector_size
embedding_matrix = np.zeros((VOCAB_SIZE, embedding_dim))
# Creating thw embedding_matrix based on W2V model in gensim
for word, i in tokenizer.word_index.items():
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]
    else:
        embedding_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_CNN():
    model = CNN(
        vocab_size=VOCAB_SIZE,
        embedding_dim=cnn_config["embedding_dim"],
        num_filters=cnn_config["num_filters"],
        filter_sizes=cnn_config["filter_sizes"],
        dropout=cnn_config["dropout_rate"],
        embedding_matrix=embedding_matrix,
    ).to(device)
    model.load_state_dict(torch.load("checkpoints/best_model_CNN.pt"))
    return model


def load_RNN():
    model = SentimentRNN(
        vocab_size=VOCAB_SIZE,
        embedding_dim=rnn_config["embedding_dim"],
        hidden_size=rnn_config["hidden_size"],
        tagset_size=1,
        n_layers=rnn_config["num_layers"],
        dropout_rate=rnn_config["dropout_rate"],
    ).to(device)
    model.load_state_dict(torch.load("checkpoints/best_model_RNN.pt"))
    return model


def load_GRU():
    model = GRU(
        vocab_size=VOCAB_SIZE,
        embedding_dim=gru_config["embedding_dim"],
        hidden_size=gru_config["hidden_size"],
        tagset_size=1,
        n_layers=gru_config["num_layers"],
        dropout_rate=gru_config["dropout_rate"],
    ).to(device)
    model.load_state_dict(torch.load("checkpoints/best_model_GRU.pt"))
    return model


def load_BiLSTM_Attention():
    model = BiLSTM_Attention(
        vocab_size=VOCAB_SIZE,
        embedding_dim=bilstm_config["embedding_dim"],
        hidden_size=bilstm_config["hidden_size"],
        tagset_size=1,
        n_layers=bilstm_config["num_layers"],
        dropout_rate=bilstm_config["dropout_rate"],
    ).to(device)
    model.load_state_dict(torch.load("checkpoints/best_model_BiLSTM.pt"))
    return model


def use_bert(text):
    bert_path = "./transformer/bert-train/suicide_pretrained"
    config = BertConfig.from_pretrained(bert_path, num_hidden_layers=1, num_labels=2)
    model = BertForSequenceClassification.from_pretrained(bert_path, config=config)
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    input_token = tokenizer(text, truncation=True, padding=True)
    model.eval()
    output = model(torch.tensor([input_token["input_ids"]]))
    x = output.logits[0, 1].item()
    p = math.exp(x) / (1 + math.exp(x))
    label = "Suicidal" if p > 0.5 else "Non-suicidal"
    print("Predicted sentiment: ", label, " with confidence: ", max(p, 1 - p))


def use_electra(text):
    electra_path = "./transformer/electra/finetuned-electra"
    tokenizer = ElectraTokenizerFast.from_pretrained(electra_path)
    model = ElectraForSequenceClassification.from_pretrained(electra_path)
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=128
    )
    model.eval()
    output = model(**inputs)
    x = output.logits[0, 1].item()
    p = math.exp(x) / (1 + math.exp(x))
    label = "Suicidal" if p > 0.5 else "Non-suicidal"
    print("Predicted sentiment: ", label, " with confidence: ", max(p, 1 - p))


def main():
    while True:
        model_type = input(
            "Enter the model type:\n1: CNN\n2: RNN\n3: GRU\n4: LSTM (BiLSTM_Attention)\n5: BERT\n6: ELECTRA\n"
        )
        if model_type in map(str, range(1, 7)):
            break
        else:
            print("Invalid model type. Please enter a valid model type.")

    text = ""
    res = input(
        'Enter the text to predict the sentiment: (terminate by a "###" line)\n'
    )

    while res != "###":
        text += res
        res = input()

    if model_type == "1":
        model = load_CNN()
    elif model_type == "2":
        model = load_RNN()
    elif model_type == "3":
        model = load_GRU()
    elif model_type == "4":
        model = load_BiLSTM_Attention()
    elif model_type == "5":
        use_bert(text)
        return
    elif model_type == "6":
        use_electra(text)
        return

    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=100, padding="post")
    text = torch.tensor(padded).to(device)
    output = model(text)
    output = output.cpu().detach().numpy()[0]
    label = "Suicidal" if output > 0.5 else "Non-suicidal"
    print("Predicted sentiment: ", label, " with confidence: ", max(output, 1 - output))


if __name__ == "__main__":
    main()
