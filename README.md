# Suicide Detection based on NLP-DeepLearning
Suicide Detection data : [data](https://husteduvn-my.sharepoint.com/:f:/g/personal/minh_nn225510_sis_hust_edu_vn/ElMsC1qDV-xKoBvgySROcuABbhgj_nQYGB6c5TH0pIZggQ?e=zPpgjw)

## Project Description

This project addresses suicide detection through binary text classification using a blend of 
NLP models. Text is preprocessed with both a custom-built Word2Vec model and pre-trained 
Gensim Word2Vec embeddings, alongside pre-trained Transformer models for rich contextual 
understanding. The classification task is approached with RNN and CNN architectures to 
capture sequential patterns and key features in text, enhanced by Transformers for advanced 
attention-based processing. The goal is to experiment many models including both traditional 
models and emerging models on binary classification task. This reliable detection supports 
early intervention systems, utilizing key metrics like F1 score and accuracy.

## Folder structures

```
.
├── embeddings/
|   ├── CBOW_embeddings.pkl              # trained CBOW embeddings
|   ├── SkipGram_embeddings.pkl          # trained SkipGram embeddings
|   ├──tokenizer.json                    # tokenizer dictionary
├── model/
|   ├── attention_bilstm.py              # Bi-LSTM with Attention model
|   ├── gru.py                           # GRU model
|   ├── lstm.py                          # LSTM model
|   ├── rnn.py                           # RNN model
|   └── cnn.py                           # CNN model
├── preprocessing_model/
|   ├── label_encoder_wunk.py            # super class of label encoder with dunk
|   └── w2v.py                           # W2V model(Cbow and SkipGram)
├── suicide_data.py                      # Customed data class
├── tokenizer.py                         # Build common tokenizer for RNNs and CNN model, data split
├── utils.py                             # utility functions
├── config.json                          # saved hyperparameters for models
├── notebook.ipynb                       # Experimental results of CNN and RNNs model
├── data_cleaning.py                     # Cleaning data
└── requirement.txt                      # Dependencies

```
## UI guide

Please download folder **suicide-pretrained** from [data](https://husteduvn-my.sharepoint.com/:f:/g/personal/minh_nn225510_sis_hust_edu_vn/ElMsC1qDV-xKoBvgySROcuABbhgj_nQYGB6c5TH0pIZggQ?e=zPpgjw) and place it into destination **UI/utils**.

To run the UI: type in cmd **python UI/app.py**
