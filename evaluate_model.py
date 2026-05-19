import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from her2_full_pipeline import HER2FusionNet, HER2Dataset # Import your model and dataset classes
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate(test_path, model_weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}")

    # 1. Load the Test Dataset
    test_ds = HER2Dataset(test_path)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    # 2. Initialize and Load the Saved Model
    model = HER2FusionNet(num_classes=4).to(device)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    print("Running inference on test set...")
    with torch.no_grad():
        for tiss, text, labels in test_loader:
            tiss, text, labels = tiss.to(device), text.to(device), labels.to(device)
            
            outputs = model(tiss, text)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 3. Calculate Metrics
    print("\n--- Classification Report ---")
    target_names = ['HER2-0', 'HER2-1+', 'HER2-2+', 'HER2-3+']
    print(classification_report(all_labels, all_preds, target_names=target_names))

    # 4. Generate Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('HER2FusionNet Confusion Matrix')
    plt.savefig('test_confusion_matrix.png') # This saves the visual result for your paper
    print("Confusion matrix saved as test_confusion_matrix.png")
    plt.show()

if __name__ == "__main__":
    # Update these paths
    TEST_DATA_PATH = './BCI_Patches/test'
    WEIGHTS_PATH = 'best_her2_model.pth'
    
    evaluate(TEST_DATA_PATH, WEIGHTS_PATH)
