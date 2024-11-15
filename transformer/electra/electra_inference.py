import torch
from transformers import ElectraTokenizerFast,ElectraForSequenceClassification

RANDOM_SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = ElectraTokenizerFast.from_pretrained("./finetuned-electra")
discriminator = ElectraForSequenceClassification.from_pretrained("./finetuned-electra")

def inference(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = discriminator(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return prediction #0 is benign, 1 is suicide

if __name__ == "__main__":
    print(inference(input()))