import argparse
from datetime import datetime
import pathlib
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier

from utils import DataAugmentation, Head, Loss, MultiCropWrapper, clip_gradients

def compute_embedding(backbone, data_loader, device="cuda"):
    """Compute embeddings for an entire dataset."""
    backbone.eval()
    embeddings = []
    labels = []
    images = []
    
    with torch.no_grad():
        for img, label in tqdm.tqdm(data_loader, desc="Computing embeddings"):
            img = img.to(device)
            features = backbone(img).cpu()
            embeddings.append(features)
            labels.append(label)
            images.append(img.cpu())
    
    return (
        torch.cat(embeddings, dim=0),
        torch.cat(images, dim=0),
        torch.cat(labels, dim=0)
    )

def train_single_fold(args, class_dir, fold_num, tensorboard_root, checkpoints_root):
    """Train a binary classifier for a single fold."""
    # Data preparation
    transform_aug = DataAugmentation(
        size=224,
        n_local_crops=args.n_crops - 2,
        global_crops_scale=(0.4, 1.0),
        local_crops_scale=(0.05, 0.4)
    )

    transform_plain = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Dataset setup
    fold_path = class_dir / f"fold{fold_num}"
    if not fold_path.exists():
        print(f"Skipping fold {fold_num} - directory not found")
        return

    dataset_train_aug = ImageFolder(fold_path / "train", transform=transform_aug)
    dataset_train_plain = ImageFolder(fold_path / "train", transform=transform_plain)
    dataset_val_plain = ImageFolder(fold_path / "val", transform=transform_plain)

    # DataLoaders
    data_loader_train_aug = DataLoader(
        dataset_train_aug,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )

    # Model configuration
    model_config = {
        "vit_s": {"name": "dinov2_vits14", "dim": 384},
        "vit_b": {"name": "dinov2_vitb14", "dim": 768}
    }[args.model_type]

    # Model initialization
    def create_model():
        backbone = torch.hub.load(
            "facebookresearch/dinov2",
            model_config["name"],
            pretrained=args.pretrained
        )
        return MultiCropWrapper(
            backbone,
            Head(
                model_config["dim"],
                args.out_dim,
                hidden_dim=512,
                bottleneck_dim=256,
                norm_last_layer=args.norm_last_layer
            )
        )

    student = create_model().to(args.device)
    teacher = create_model().to(args.device)
    
    # Initialize teacher with student weights
    with torch.no_grad():
        for s_param, t_param in zip(student.parameters(), teacher.parameters()):
            t_param.copy_(s_param)
            t_param.requires_grad = False

    # Loss and optimizer
    loss_inst = Loss(args.out_dim, args.teacher_temp, args.student_temp).to(args.device)
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=0.0005 * args.batch_size / 256,
        weight_decay=args.weight_decay
    )

    # Create save directories
    class_name = class_dir.name
    save_dir = checkpoints_root / class_name / f"fold{fold_num}"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = save_dir / "best_model.pth"
    final_model_path = save_dir / "final_model.pth"

    # Training setup
    writer = SummaryWriter(tensorboard_root / f"fold{fold_num}")
    best_f1 = 0
    steps_since_best = 0
    n_steps = 0

    try:
        for epoch in range(args.n_epochs):
            for batch_idx, (images, _) in tqdm.tqdm(
                enumerate(data_loader_train_aug),
                total=len(data_loader_train_aug),
                desc=f"Epoch {epoch+1}/{args.n_epochs}"
            ):
                student.train()
                
                # Forward pass
                images = [img.to(args.device) for img in images]
                teacher_output = teacher(images[:2])  # Global crops
                student_output = student(images)
                
                # Compute loss
                loss = loss_inst(student_output, teacher_output)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                clip_gradients(student, args.clip_grad)
                optimizer.step()

                # EMA update for teacher
                with torch.no_grad():
                    for s_param, t_param in zip(student.parameters(), teacher.parameters()):
                        t_param.mul_(args.momentum_teacher)
                        t_param.add_((1 - args.momentum_teacher) * s_param.detach())

                # Evaluation and logging
                if n_steps % args.logging_freq == 0:
                    student.eval()
                    
                    # Compute embeddings
                    train_embs, _, train_labels = compute_embedding(
                        student.backbone, 
                        DataLoader(
                            dataset_train_plain,
                            batch_size=args.batch_size_eval,
                            shuffle=False,
                            num_workers=2
                        ),
                        device=args.device
                    )
                    
                    val_embs, _, val_labels = compute_embedding(
                        student.backbone,
                        DataLoader(
                            dataset_val_plain,
                            batch_size=args.batch_size_eval,
                            shuffle=False,
                            num_workers=2
                        ),
                        device=args.device
                    )
                    
                    # Calculate F1 score
                    knn = KNeighborsClassifier(n_neighbors=20)
                    knn.fit(train_embs.numpy(), train_labels.numpy())
                    val_pred = knn.predict(val_embs.numpy())
                    current_f1 = f1_score(val_labels.numpy(), val_pred, pos_label=1)

                    # Early stopping and model saving
                    if current_f1 > best_f1:
                        torch.save(student.state_dict(), best_model_path)
                        best_f1 = current_f1
                        steps_since_best = 0
                    else:
                        steps_since_best += 1
                        if steps_since_best >= args.early_stop_patience:
                            print(f"Early stopping at step {n_steps} | Best F1: {best_f1:.4f}")
                            break

                    # TensorBoard logging
                    writer.add_scalar("train_loss", loss.item(), n_steps)
                    writer.add_scalar("f1-score-positive", current_f1, n_steps)
                    
                    student.train()

                n_steps += 1

    finally:
        # Save final model state
        torch.save(student.state_dict(), final_model_path)
        print(f"Saved final model to: {final_model_path}")
        writer.close()

def main():
    parser = argparse.ArgumentParser(
        "DINO Training with ViT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Training parameters
    parser.add_argument("-b", "--batch-size", type=int, default=32)
    parser.add_argument("-d", "--device", type=str, choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("-l", "--logging-freq", type=int, default=200)
    parser.add_argument("--momentum-teacher", type=float, default=0.9995)
    parser.add_argument("-c", "--n-crops", type=int, default=4)
    parser.add_argument("-e", "--n-epochs", type=int, default=20)
    parser.add_argument("-o", "--out-dim", type=int, default=256)
    parser.add_argument("-t", "--tensorboard-dir", type=str, default="logs")
    parser.add_argument("--clip-grad", type=float, default=2.0)
    parser.add_argument("--norm-last-layer", action="store_true")
    parser.add_argument("--batch-size-eval", type=int, default=64)
    parser.add_argument("--teacher-temp", type=float, default=0.04)
    parser.add_argument("--student-temp", type=float, default=0.1)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("-w", "--weight-decay", type=float, default=0.4)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--model-type", choices=["vit_s", "vit_b"], default="vit_b")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints",
                       help="Root directory for saving model checkpoints")
    
    args = parser.parse_args()
    print("Global training configuration:")
    print(vars(args))

    # Create root directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoints_root = pathlib.Path(args.checkpoints_dir) / timestamp
    checkpoints_root.mkdir(parents=True, exist_ok=True)
    
    # Save config file
    with open(checkpoints_root / "config.txt", "w") as f:
        f.write("Training configuration:\n")
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    # Process each class and each fold
    fsl_root = pathlib.Path("/home/alexsandro/pgcc/data/mestrado_Alexsandro/cross_validation/fsl")
    class_dirs = sorted([d for d in fsl_root.iterdir() if d.is_dir()])

    for class_dir in class_dirs:
        class_name = class_dir.name
        print(f"\n{'='*40}")
        print(f"Processing class: {class_name}")
        print(f"{'='*40}")

        # Create class-specific log directory
        tensorboard_dir = pathlib.Path(args.tensorboard_dir) / timestamp / class_name
        tensorboard_dir.mkdir(parents=True, exist_ok=True)

        # Process all 5 folds
        for fold_num in range(5):
            print(f"\n{'='*20} Fold {fold_num} {'='*20}")
            train_single_fold(
                args,
                class_dir,
                fold_num,
                tensorboard_dir,
                checkpoints_root
            )

if __name__ == "__main__":
    main()