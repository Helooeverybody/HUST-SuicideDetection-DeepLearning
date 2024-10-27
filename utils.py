import torch
import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def map_to_labels(pred):
    return (pred > 0.5).to(device)

def accuracy_cal(pred,labels):
    labels=labels.float()
    mapped_pred=map_to_labels(pred)
    correct_num=(labels==mapped_pred).sum().item()
    acc=correct_num/len(pred)
    return acc

def precision_recall_f1(preds, labels):
    # Initialize counts
    TP_total, FP_total, FN_total = 0, 0, 0
    labels=labels.float()
    mapped_preds=map_to_labels(preds)
    # Iterate over batches
    for preds, labels in zip(mapped_preds, labels):
        # True Positives (TP)
        TP = ((preds == 1) & (labels == 1)).sum().item()
        # False Positives (FP)
        FP = ((preds == 1) & (labels == 0)).sum().item()
        # False Negatives (FN)
        FN = ((preds == 0) & (labels == 1)).sum().item()
        # Accumulate counts
        TP_total += TP
        FP_total += FP
        FN_total += FN
    # Precision: TP / (TP + FP)
    precision = TP_total / (TP_total + FP_total) if (TP_total + FP_total) > 0 else 0
    # Recall: TP / (TP + FN)
    recall = TP_total / (TP_total + FN_total) if (TP_total + FN_total) > 0 else 0
    # F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return [precision, recall, f1]


def sketch(name, train_loss, val_loss, train_acc, val_acc, train_f1, val_f1):
    epochs = range(1, len(train_loss) + 1)  # Assume the length of the loss lists represents the number of epochs
    
    # Initialize a figure with subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(name)

    # Plot training and validation loss
    axs[0].plot(epochs, train_loss, label="Train Loss", color='blue', marker='o')
    axs[0].plot(epochs, val_loss, label="Validation Loss", color='orange', marker='o')
    axs[0].set_title("Loss")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    
    # Plot training and validation accuracy
    axs[1].plot(epochs, train_acc, label="Train Accuracy", color='blue', marker='o')
    axs[1].plot(epochs, val_acc, label="Validation Accuracy", color='orange', marker='o')
    axs[1].set_title("Accuracy")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()
    
    # Plot training and validation F1 score
    axs[2].plot(epochs, train_f1, label="Train F1 Score", color='blue', marker='o')
    axs[2].plot(epochs, val_f1, label="Validation F1 Score", color='orange', marker='o')
    axs[2].set_title("F1 Score")
    axs[2].set_xlabel("Epochs")
    axs[2].set_ylabel("F1 Score")
    axs[2].legend()
    
    # Adjust layout and show plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
