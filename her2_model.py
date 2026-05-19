import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# ==========================================
# 1. MODEL COMPONENTS (Architecture)
# ==========================================

class LearnableGaborConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=3):
        super(LearnableGaborConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
    def forward(self, x):
        return self.conv(x)

class TissueNet(nn.Module):
    def __init__(self):
        super(TissueNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 11, padding=5), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 7, padding=3), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 7, padding=3), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 5, padding=2), nn.BatchNorm2d(128), nn.ReLU(True)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
    def forward(self, x):
        return self.adaptive_pool(self.features(x))

class TextureNet(nn.Module):
    def __init__(self):
        super(TextureNet, self).__init__()
        self.features = nn.Sequential(
            LearnableGaborConv2d(3, 32, 7, padding=3),
            LearnableGaborConv2d(32, 32, 7, padding=3),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((64, 64))
    def forward(self, x):
        return self.adaptive_pool(self.features(x))

class CrossAttentionFusion(nn.Module):
    def __init__(self, c1=128, c2=256, h2=64, w2=64):
        super(CrossAttentionFusion, self).__init__()
        self.c1, self.h2, self.w2, self.num_heads = c1, h2, w2, 3
        self.channel_alignment = nn.Conv2d(c2, c1, kernel_size=1)
        self.spatial_alignment = nn.ConvTranspose2d(c1, c1, kernel_size=8, stride=8)
        self.d_head = math.ceil(c1 / self.num_heads)
        self.project_dim = self.num_heads * self.d_head
        self.W_q = nn.Linear(c1, self.project_dim)
        self.W_k = nn.Linear(c1, self.project_dim)
        self.W_v = nn.Linear(c1, self.project_dim)
        self.W_o = nn.Linear(self.project_dim, c1)

    def forward(self, f1, f2):
        B = f1.shape[0]
        f2_p = self.channel_alignment(f2)
        f1_p = self.spatial_alignment(f1)
        f1_flat = f1_p.permute(0, 2, 3, 1).contiguous().view(B, -1, self.c1)
        f2_flat = f2_p.permute(0, 2, 3, 1).contiguous().view(B, -1, self.c1)
        q, k, v = self.W_q(f1_flat), self.W_k(f2_flat), self.W_v(f2_flat)
        q = q.view(B, -1, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.d_head).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        f_a = torch.matmul(F.softmax(attn, dim=-1), v)
        f_a = f_a.transpose(1, 2).contiguous().view(B, -1, self.project_dim)
        return self.W_o(f_a).view(B, self.h2, self.w2, self.c1).permute(0, 3, 1, 2)

# ==========================================
# 2. CLASS SPECIFIC HEADS
# ==========================================

class HER2FusionNet(nn.Module):
    def __init__(self, num_classes=4):
        super(HER2FusionNet, self).__init__()
        self.tissue_net = TissueNet()
        self.texture_net = TextureNet()
        self.fusion_module = CrossAttentionFusion()
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # --- Specialized Classification Heads ---
        # Instead of one layer with 4 outputs, we create 4 heads with 1 output each
        # This matches: z_c = W_c * f + b_c for c = 1, 2, 3, 4
        self.head_0 = nn.Linear(128, 1) # Specialized for HER2-0
        self.head_1 = nn.Linear(128, 1) # Specialized for HER2-1+
        self.head_2 = nn.Linear(128, 1) # Specialized for HER2-2+
        self.head_3 = nn.Linear(128, 1) # Specialized for HER2-3+

    def forward(self, img_tiss, img_text):
        # 1. Feature Extraction
        f1 = self.tissue_net(img_tiss)
        f2 = self.texture_net(img_text)
        
        # 2. Fusion via Cross-Attention
        f_fused = self.fusion_module(f1, f2)
        
        # 3. Global Pooling to get shared feature vector 'f'
        f = self.global_avg_pool(f_fused).view(f_fused.size(0), -1) 
        
        # 4. Apply 4 Specialized Heads independently
        z0 = self.head_0(f) # Logit for Class 0
        z1 = self.head_1(f) # Logit for Class 1
        z2 = self.head_2(f) # Logit for Class 2
        z3 = self.head_3(f) # Logit for Class 3
        
        # 5. Concatenate logits to form z = [z0, z1, z2, z3]
        logits = torch.cat([z0, z1, z2, z3], dim=1)
        
        # Note: Softmax is typically handled by nn.CrossEntropyLoss during training
        return logits
# ==========================================
# 3. DATASET MANAGEMENT
# ==========================================

class HER2Dataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        # Ensure these names match your actual folder names exactly
        self.classes = ['HER2_0', 'HER2_1', 'HER2_2', 'HER2_3']
        self.image_paths = []
        self.labels = []

        for idx, cls in enumerate(self.classes):
            cls_folder = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_folder): continue
            for img_name in os.listdir(cls_folder):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(cls_folder, img_name))
                    self.labels.append(idx)

        self.tiss_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.text_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        return self.tiss_transform(img), self.text_transform(img), self.labels[idx]

# ==========================================
# 4. TRAINING & SAVING LOGIC
# ==========================================

def train_model(train_path, val_path, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Loaders
    train_loader = DataLoader(HER2Dataset(train_path), batch_size=16, shuffle=True)
    val_loader = DataLoader(HER2Dataset(val_path), batch_size=16, shuffle=False)

    model = HER2FusionNet(num_classes=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for tiss, text, labels in train_loader:
            tiss, text, labels = tiss.to(device), text.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(tiss, text)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        # Simple Validation Check
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for tiss, text, labels in val_loader:
                tiss, text, labels = tiss.to(device), text.to(device), labels.to(device)
                outputs = model(tiss, text)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {running_loss/len(train_loader):.4f} - Val Acc: {val_acc:.2f}%")

        # Save the BEST model automatically
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_her2_model.pth')
            print("--> Best Model Saved!")

if __name__ == "__main__":
    # CHANGE THESE to your folders before running
    TRAIN_FOLDER = '/content/drive/MyDrive/Data/train'
    VAL_FOLDER = '/content/drive/MyDrive/Data/val'
    
    if os.path.exists(TRAIN_FOLDER):
        train_model(TRAIN_FOLDER, VAL_FOLDER)
    else:
        print("Folder not found. Please check TRAIN_FOLDER path.")
