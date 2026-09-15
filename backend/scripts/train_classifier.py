"""Train lightweight classifier head on frozen IndicWav2Vec embeddings.

Usage:
  python scripts/build_manifest.py
  python scripts/train_classifier.py --manifest ../data/manifest.csv

Requires manifest.csv with columns: filepath, label, language
  label: 0=genuine, 1=cloned
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.indic_encoder import ClassifierHead, extract_embedding, load_encoder  # noqa: E402

OUT_DIR = ROOT / "models" / "indic_classifier"
EMB_CACHE = OUT_DIR / "embeddings_cache.npz"


def load_manifest(path: Path) -> list[dict]:
    import csv

    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "filepath": row["filepath"],
                "label": int(row["label"]),
                "language": row.get("language", "unknown"),
            })
    return rows


def load_audio(path: str):
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav, sr


def extract_all_embeddings(rows: list[dict], use_cache: bool = True):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if use_cache and EMB_CACHE.exists():
        data = np.load(EMB_CACHE, allow_pickle=True)
        print(f"[INFO] Loaded cached embeddings from {EMB_CACHE}")
        return data["X"], data["y"], data["languages"].tolist(), data["paths"].tolist()

    load_encoder()
    X, y, languages, paths = [], [], [], []
    failed = 0
    t0 = time.time()
    for i, row in enumerate(rows):
        try:
            wav, sr = load_audio(row["filepath"])
            emb = extract_embedding(wav, sr)
            if emb is None:
                failed += 1
                continue
            X.append(emb.numpy())
            y.append(row["label"])
            languages.append(row["language"])
            paths.append(row["filepath"])
        except Exception as e:
            failed += 1
            print(f"[FAIL] {row['filepath']}: {e}")
        if (i + 1) % 10 == 0:
            print(f"  extracted {i+1}/{len(rows)} ({failed} failed)")

    X = np.stack(X)
    y = np.array(y)
    print(f"[OK] Extracted {len(y)} embeddings in {time.time()-t0:.1f}s ({failed} failed)")
    np.savez(EMB_CACHE, X=X, y=y, languages=np.array(languages), paths=np.array(paths))
    return X, y, languages, paths


def check_leakage(train_paths, test_paths):
    train_set = set(train_paths)
    test_set = set(test_paths)
    overlap = train_set & test_set
    if overlap:
        print(f"[LEAKAGE WARNING] {len(overlap)} files appear in BOTH train and test!")
        for p in list(overlap)[:5]:
            print(f"  {p}")
        return True
    print("[OK] No filepath overlap between train and test sets")
    return False


def per_language_metrics(y_true, y_pred, languages):
    from collections import defaultdict

    metrics = {}
    by_lang = defaultdict(lambda: {"y": [], "p": []})
    for yt, yp, lang in zip(y_true, y_pred, languages):
        by_lang[lang]["y"].append(yt)
        by_lang[lang]["p"].append(yp)
    for lang, d in sorted(by_lang.items()):
        if len(d["y"]) < 2:
            continue
        metrics[lang] = {
            "n": len(d["y"]),
            "accuracy": round(accuracy_score(d["y"], d["p"]) * 100, 1),
            "f1": round(f1_score(d["y"], d["p"], zero_division=0), 3),
        }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT.parent / "data" / "manifest.csv")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"[ERROR] Manifest not found: {args.manifest}")
        print("Run: python scripts/build_manifest.py first")
        sys.exit(1)

    rows = load_manifest(args.manifest)
    print(f"Manifest: {len(rows)} samples from {args.manifest}")
    if len(rows) < 20:
        print("[WARN] Very small dataset — metrics will NOT be statistically meaningful")

    X, y, languages, paths = extract_all_embeddings(rows, use_cache=not args.no_cache)
    input_dim = X.shape[1]

    # Stratified split by label AND language (falls back to label-only if too small)
    stratify = [f"{l}_{lang}" for l, lang in zip(y, languages)]
    from collections import Counter
    min_class = min(Counter(stratify).values())
    if min_class < 2:
        print(f"[WARN] Cannot stratify by language+label (min class size={min_class}). Using label-only split.")
        stratify_labels = y
    else:
        stratify_labels = stratify

    X_train, X_test, y_train, y_test, lang_train, lang_test, path_train, path_test = train_test_split(
        X, y, languages, paths,
        test_size=0.25,
        random_state=42,
        stratify=stratify_labels,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train,
    )

    print(f"\nSplit: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")
    check_leakage(path_train, path_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClassifierHead(input_dim, args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
        batch_size=32, shuffle=True,
    )

    best_val_f1 = 0.0
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        model.eval()
        with torch.inference_mode():
            val_logits = model(torch.tensor(X_val, dtype=torch.float32).to(device))
            val_pred = val_logits.argmax(dim=1).cpu().numpy()
        val_f1 = f1_score(y_val, val_pred, zero_division=0)
        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:3d}  loss={total_loss/len(train_loader):.4f}  val_f1={val_f1:.3f}")

    model.load_state_dict(best_state)
    model.eval()

    with torch.inference_mode():
        test_logits = model(torch.tensor(X_test, dtype=torch.float32).to(device))
        y_pred = test_logits.argmax(dim=1).cpu().numpy()

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 60)
    print("HELD-OUT TEST RESULTS")
    print("=" * 60)
    print(f"Accuracy:  {acc*100:.1f}%")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1:        {f1:.3f}")
    print(f"\nConfusion matrix (rows=true, cols=pred) [genuine=0, cloned=1]:")
    print(cm)
    print(f"\n{classification_report(y_test, y_pred, target_names=['genuine','cloned'])}")

    lang_metrics = per_language_metrics(y_test, y_pred, lang_test)
    print("Per-language test breakdown:")
    for lang, m in lang_metrics.items():
        print(f"  {lang}: n={m['n']}, acc={m['accuracy']}%, f1={m['f1']}")

    if acc > 0.97:
        print("\n[FLAG] Accuracy >97% — verify no train/test leakage before presenting!")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, OUT_DIR / "classifier_head.pt")
    cfg = {
        "input_dim": input_dim,
        "hidden": args.hidden,
        "metrics": {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1},
        "per_language": lang_metrics,
        "train_size": len(y_train),
        "test_size": len(y_test),
    }
    with open(OUT_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    with open(OUT_DIR / "test_predictions.npz", "wb") as f:
        np.savez(f, y_true=y_test, y_pred=y_pred, y_prob=torch.softmax(test_logits, dim=-1)[:, 1].cpu().numpy())

    print(f"\n[OK] Saved head to {OUT_DIR / 'classifier_head.pt'}")


if __name__ == "__main__":
    main()
