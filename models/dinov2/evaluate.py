import argparse
import json
import pathlib
import numpy as np
import torch
import torchvision.transforms as transforms
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

# Recreate necessary components from training script
class MultiCropWrapper(torch.nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.new_head = head  # Changed name to match checkpoint key

    def forward(self, x):
        return self.new_head(self.backbone(x))

class Head(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=512, bottleneck_dim=256, norm_last_layer=True):
        super().__init__()
        self.mlp = torch.nn.Sequential(  # Changed name to match checkpoint
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, bottleneck_dim),
            torch.nn.LayerNorm(bottleneck_dim),
        )
        self.last_layer = torch.nn.utils.weight_norm(  # Changed structure
            torch.nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        if norm_last_layer:
            self.last_layer.weight_g.data.fill_(1)  # Match training configuration

    def forward(self, x):
        x = self.mlp(x)
        x = self.last_layer(x)
        return x

def compute_embeddings(model, data_loader, device="cuda"):
    """Compute embeddings using model backbone"""
    model.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for img, label in tqdm(data_loader, desc="Computing embeddings"):
            img = img.to(device)
            features = model.backbone(img).cpu()
            embeddings.append(features)
            labels.append(label)
    
    return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0)

def evaluate_fold(checkpoint_path, data_path, model_type, device="cuda"):
    """Evaluate a single fold's best model"""
    # Load model configuration
    model_config = {
        "vit_s": {"name": "dinov2_vits14", "dim": 384},
        "vit_b": {"name": "dinov2_vitb14", "dim": 768}
    }[model_type]

    # Recreate model architecture
    backbone = torch.hub.load("facebookresearch/dinov2", model_config["name"])
    head = Head(
        model_config["dim"],
        256,  # Should match training's out_dim
        hidden_dim=512,
        bottleneck_dim=256,
        norm_last_layer=True
    )
    model = MultiCropWrapper(backbone, head).to(device)
    
    # Load trained weights
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"\nLoaded model from {checkpoint_path}")

    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Load datasets
    train_dataset = ImageFolder(data_path / "train", transform=transform)
    val_dataset = ImageFolder(data_path / "val", transform=transform)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2
    )

    # Compute embeddings
    train_embs, train_labels = compute_embeddings(model, train_loader, device)
    val_embs, val_labels = compute_embeddings(model, val_loader, device)

    # Convert to numpy arrays
    X_train = train_embs.numpy()
    y_train = train_labels.numpy()
    X_val = val_embs.numpy()
    y_val = val_labels.numpy()

    # Train KNN classifier
    knn = KNeighborsClassifier(n_neighbors=20)
    knn.fit(X_train, y_train)
    
    # Get predictions and probabilities
    y_pred = knn.predict(X_val)
    y_probs = knn.predict_proba(X_val)[:, 1]  # Probability of positive class

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, pos_label=1),
        "recall": recall_score(y_val, y_pred, pos_label=1),
        "f1": f1_score(y_val, y_pred, pos_label=1),
        "auc": roc_auc_score(y_val, y_probs)
    }
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--checkpoints-dir", type=str, required=True,
                       help="Root directory containing trained models")
    parser.add_argument("--data-root", type=str, required=True,
                       help="Path to cross_validation/fsl directory")
    parser.add_argument("--model-type", choices=["vit_s", "vit_b"], required=True,
                       help="Model architecture used during training")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"], help="Device for computation")
    
    args = parser.parse_args()

    checkpoints_root = pathlib.Path(args.checkpoints_dir)
    data_root = pathlib.Path(args.data_root)

    # Process all classes and folds
    results = {}
    
    for class_dir in sorted(data_root.iterdir()):
        if not class_dir.is_dir():
            continue
            
        class_name = class_dir.name
        print(f"\n{'='*40}")
        print(f"Evaluating class: {class_name}")
        print(f"{'='*40}")
        
        class_results = []
        
        for fold_num in range(5):
            fold_dir = checkpoints_root / class_name / f"fold{fold_num}"
            checkpoint_path = fold_dir / "best_model.pth"
            data_path = class_dir / f"fold{fold_num}"
            
            if not checkpoint_path.exists():
                print(f"Skipping fold {fold_num} - no checkpoint found")
                continue
                
            print(f"\nEvaluating fold {fold_num}")
            metrics = evaluate_fold(
                checkpoint_path=checkpoint_path,
                data_path=data_path,
                model_type=args.model_type,
                device=args.device
            )
            
            class_results.append(metrics)
            print("\nFold metrics:")
            for k, v in metrics.items():
                print(f"{k:>10}: {v:.4f}")
                
            # Save fold metrics
            with open(fold_dir / "evaluation_metrics.json", "w") as f:
                json.dump(metrics, f, indent=4)

        # Calculate class-level aggregates
        if class_results:
            aggregated = {}
            for metric in ["accuracy", "precision", "recall", "f1", "auc"]:
                values = [r[metric] for r in class_results]
                aggregated[f"{metric}_mean"] = np.mean(values)
                aggregated[f"{metric}_std"] = np.std(values)
            
            results[class_name] = aggregated
            
            # Save class aggregates
            class_save_path = checkpoints_root / class_name / "aggregated_metrics.json"
            with open(class_save_path, "w") as f:
                json.dump(aggregated, f, indent=4)
                
            print(f"\nAggregated metrics for {class_name}:")
            for k, v in aggregated.items():
                print(f"{k:>15}: {v:.4f} ± {aggregated[k.replace('_mean', '_std')]:.4f}")

    # Save final summary
    summary_path = checkpoints_root / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved final evaluation summary to {summary_path}")

if __name__ == "__main__":
    main()