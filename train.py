import os, json
import torch
import torch.nn as nn
import multiprocessing as mp
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import CNNCarDamage

def main():
    TRAIN_DIR = "dataset/train"
    VAL_DIR   = "dataset/val"
    TEST_DIR  = "dataset/test"

    IMG_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 15
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    SAVE_PATH = "car_damage_cnn.pth"
    META_PATH = "car_damage_meta.json"

    train_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tfms)
    val_ds   = datasets.ImageFolder(VAL_DIR,   transform=eval_tfms)
    test_ds  = datasets.ImageFolder(TEST_DIR,  transform=eval_tfms)

    num_workers = 0
    pin_memory = (DEVICE == "cuda")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    class_to_idx = train_ds.class_to_idx
    idx_to_class = {v:k for k,v in class_to_idx.items()}

    def find_positive_class_idx(idx_to_class):
        for idx, name in idx_to_class.items():
            n = name.lower()
            if "damag" in n or "hasar" in n:
                return idx
        return 1

    pos_class_idx = find_positive_class_idx(idx_to_class)

    def to_binary_target(y):
        return (y == pos_class_idx).float().unsqueeze(1)

    counts = [0, 0]
    for _, y in train_ds.samples:
        counts[int(y == pos_class_idx)] += 1  # [neg, pos]
    neg_count, pos_count = counts[0], counts[1]
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)], device=DEVICE)

    model = CNNCarDamage().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    @torch.no_grad()
    def evaluate(loader, threshold=0.5):
        model.eval()
        total = 0
        correct = 0
        loss_sum = 0.0
        tp = fp = tn = fn = 0

        for x, y_raw in loader:
            x = x.to(DEVICE)
            y = to_binary_target(y_raw.to(DEVICE))
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += loss.item() * x.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()

            correct += (preds == y).sum().item()
            total += y.numel()

            tp += ((preds == 1) & (y == 1)).sum().item()
            fp += ((preds == 1) & (y == 0)).sum().item()
            tn += ((preds == 0) & (y == 0)).sum().item()
            fn += ((preds == 0) & (y == 1)).sum().item()

        acc = correct / total if total else 0.0
        return loss_sum / (total if total else 1), acc, {"tp": tp, "fp": fp, "tn": tn, "fn": fn}

    def metrics_from_cm(cm):
        tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
        return precision, recall, f1

    @torch.no_grad()
    def find_best_threshold(loader, t_min=0.10, t_max=0.90, step=0.01):
        best = {"threshold": 0.5, "f1": -1.0, "precision": 0.0, "recall": 0.0, "acc": 0.0, "cm": None}
        t = t_min
        while t <= t_max + 1e-9:
            _, acc, cm = evaluate(loader, threshold=t)
            precision, recall, f1 = metrics_from_cm(cm)
            if f1 > best["f1"]:
                best.update({"threshold": round(t, 2), "f1": f1, "precision": precision, "recall": recall, "acc": acc,
                             "cm": cm})
            t += step
        return best

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y_raw in train_loader:
            x = x.to(DEVICE)
            y = to_binary_target(y_raw.to(DEVICE))

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()

        val_loss, val_acc, _ = evaluate(val_loader)
        print(f"Epoch {epoch:02d} | val_loss={val_loss:.4f} | val_acc={val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            with open(META_PATH, "w", encoding="utf-8") as f:
                json.dump({
                    "img_size": IMG_SIZE,
                    "class_to_idx": class_to_idx,
                    "idx_to_class": idx_to_class,
                    "positive_class_idx": pos_class_idx,
                    "positive_class_name": idx_to_class[pos_class_idx]
                }, f, ensure_ascii=False, indent=2)
            print(f"✅ Saved best model (val_acc={best_val_acc*100:.2f}%)")

    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))

    test_loss_05, test_acc_05, cm_05 = evaluate(test_loader, threshold=0.5)
    p05, r05, f105 = metrics_from_cm(cm_05)
    print(
        f"TEST@0.50 | loss={test_loss_05:.4f} | acc={test_acc_05 * 100:.2f}% | P={p05 * 100:.2f}% R={r05 * 100:.2f}% F1={f105 * 100:.2f}% | CM={cm_05}")

    best = find_best_threshold(val_loader, t_min=0.10, t_max=0.90, step=0.01)
    print(
        f"BEST TH (VAL) = {best['threshold']} | acc={best['acc'] * 100:.2f}% | P={best['precision'] * 100:.2f}% R={best['recall'] * 100:.2f}% F1={best['f1'] * 100:.2f}% | CM={best['cm']}")

    test_loss_bt, test_acc_bt, cm_bt = evaluate(test_loader, threshold=best["threshold"])
    pbt, rbt, f1bt = metrics_from_cm(cm_bt)
    print(
        f"TEST@{best['threshold']:.2f} | loss={test_loss_bt:.4f} | acc={test_acc_bt * 100:.2f}% | P={pbt * 100:.2f}% R={rbt * 100:.2f}% F1={f1bt * 100:.2f}% | CM={cm_bt}")

if __name__ == "__main__":
    mp.freeze_support()
    main()
