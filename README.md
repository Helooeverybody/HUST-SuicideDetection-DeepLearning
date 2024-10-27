# HUST-SuicideDetection-DeepLearning

Suicide Detection data : [data](https://husteduvn-my.sharepoint.com/:f:/g/personal/minh_nn225510_sis_hust_edu_vn/ElMsC1qDV-xKoBvgySROcuABbhgj_nQYGB6c5TH0pIZggQ?e=zPpgjw)
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
├── config.json                           # saved hyperparameters for models
├── notebook.ipynb                        # Experimental results of CNN and RNNs model
├── data_cleaning.py                      #Cleaning data
└── requirement.txt                      # Dependencies

```
