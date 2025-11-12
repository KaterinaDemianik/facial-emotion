from tqdm import tqdm
import argparse, os, time
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from skimage.feature import hog

import timm


CLASSES = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

def get_dataloaders(root, img_size=48, batch_size=64, for_vit=False):
    size = 224 if for_vit else img_size
    tfm_train = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    tfm_eval = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    train_ds = datasets.ImageFolder(os.path.join(root, 'train'), transform=tfm_train)
    val_ds   = datasets.ImageFolder(os.path.join(root, 'validation'), transform=tfm_eval)
    test_ds  = datasets.ImageFolder(os.path.join(root, 'test'), transform=tfm_eval)

    use_pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=use_pin_memory)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=use_pin_memory)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=use_pin_memory)

    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds

# ------------------------
# ПРОСТИЙ РІВЕНЬ: HOG + LogisticRegression (sklearn)
# ------------------------
def run_simple(root):
    print("[Simple] HOG + LogisticRegression")
    def load_split(split_dir, max_per_class=500): 
        X, y = [], []
        for ci, cls in enumerate(CLASSES):
            cls_dir = Path(split_dir) / cls
            cnt = 0
            for p in cls_dir.glob("*.*"):
                img = Image.open(p).convert("L").resize((48,48))
                arr = np.array(img)
                feat = hog(arr, pixels_per_cell=(8,8), cells_per_block=(2,2), orientations=9, block_norm='L2-Hys')
                X.append(feat)
                y.append(ci)
                cnt += 1
                if cnt >= max_per_class:
                    break
        return np.array(X), np.array(y)

    X_train, y_train = load_split(os.path.join(root, "train"))
    X_val,   y_val   = load_split(os.path.join(root, "validation"), max_per_class=100)
    X_test,  y_test  = load_split(os.path.join(root, "test"),       max_per_class=100)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    clf = LogisticRegression(max_iter=4000, n_jobs=-1, verbose=0)
    clf.fit(X_train, y_train)
    val_pred = clf.predict(X_val)
    test_pred = clf.predict(X_test)

    print("\n[Validation]")
    print(classification_report(y_val, val_pred, target_names=CLASSES, digits=3))
    print("[Test]")
    print(classification_report(y_test, test_pred, target_names=CLASSES, digits=3))
    print(f"Val Acc: {accuracy_score(y_val, val_pred):.4f}, Test Acc: {accuracy_score(y_test, test_pred):.4f}")

# ------------------------
# СЕРЕДНІЙ РІВЕНЬ: невеликий CNN (PyTorch)
# ------------------------
class SmallCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), 
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128*6*6, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.net(x)

def train_torch(model, train_loader, val_loader, device, epochs=8, lr=1e-3):
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val = -1
    for ep in range(1, epochs+1):
        model.train()
        running = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            running += loss.item()*xb.size(0)
        train_loss = running/len(train_loader.dataset)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                pred = logits.argmax(1)
                correct += (pred==yb).sum().item()
                total += yb.size(0)
        val_acc = correct/total
        print(f"Epoch {ep}: train_loss={train_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), "best_cnn.pt")
    print(f"Best val_acc={best_val:.4f}")

def evaluate_torch(model, loader, device):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            preds += logits.argmax(1).cpu().tolist()
            gts   += yb.tolist()
    print(classification_report(gts, preds, target_names=CLASSES, digits=3))
    print(f"Test Acc: {accuracy_score(gts, preds):.4f}")

def predict_image_cnn(image_path, model_path="best_cnn.pt"):
    from PIL import Image
    import torch
    from torchvision import transforms

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = SmallCNN(num_classes=len(CLASSES))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(1).item()
        probs = torch.nn.functional.softmax(logits, dim=1)[0]

    print(f"✅ Predicted emotion: {CLASSES[pred]}")
    print("Top-3 probabilities:")
    top3 = torch.topk(probs, 3)
    for i in range(3):
        idx = top3.indices[i].item()
        val = top3.values[i].item()
        print(f"  {CLASSES[idx]}: {val:.3f}")
    

def run_cnn(root):
    print("[Medium] Small CNN")
    train_loader, val_loader, test_loader, *_ = get_dataloaders(root, img_size=48, batch_size=128, for_vit=False)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model = SmallCNN(num_classes=7)
    train_torch(model, train_loader, val_loader, device, epochs=10, lr=1e-3)
    model.load_state_dict(torch.load("best_cnn.pt", map_location=device))
    evaluate_torch(model.to(device), test_loader, device)



# ------------------------
# СКЛАДНИЙ РІВЕНЬ: Vision Transformer (timm, fine-tune)
# ------------------------
def run_vit(root):
    print("[Advanced] ViT fine-tuning (timm)")
    train_loader, val_loader, test_loader, *_ = get_dataloaders(root, img_size=48, batch_size=64, for_vit=True)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=7, in_chans=1)
    model = model.to(device)

    for name, p in model.named_parameters():
        if "head" not in name:
            p.requires_grad = False

    opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val = -1
    epochs = 1
    for ep in range(1, epochs+1):
        model.train()
        running = 0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {ep}"):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            running += loss.item()*xb.size(0)
        train_loss = running/len(train_loader.dataset)

        if ep == 2:
            print("Unfreezing all layers for full fine-tuning...")
            for p in model.parameters():
                p.requires_grad = True
            opt = optim.AdamW(model.parameters(), lr=5e-5)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(1)
                correct += (pred==yb).sum().item()
                total += yb.size(0)
        val_acc = correct/total
        print(f"Epoch {ep}: train_loss={train_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), "best_vit.pt")
    print(f"Best val_acc={best_val:.4f}")

    model.load_state_dict(torch.load("best_vit.pt", map_location=device))
    preds, gts = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds += logits.argmax(1).cpu().tolist()
            gts   += yb.tolist()
    print(classification_report(gts, preds, target_names=CLASSES, digits=3))
    print(f"Test Acc: {accuracy_score(gts, preds):.4f}")

# ------------------------
# entrypoint
# ------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/fer2013", help="шлях до папки з train/val/test")
    ap.add_argument("--model", type=str, choices=["simple","cnn","vit"], required=True)
    args = ap.parse_args()

    if not os.path.isdir(args.data_root):
        raise SystemExit(f"Не знайдено {args.data_root}. Переконайся, що розпаковано data/fer2013/train ...")

    t0 = time.time()
    if args.model == "simple":
        run_simple(args.data_root)
    elif args.model == "cnn":
        run_cnn(args.data_root)
    else:
        run_vit(args.data_root)
    print(f"\nГотово за {time.time()-t0:.1f} c.")