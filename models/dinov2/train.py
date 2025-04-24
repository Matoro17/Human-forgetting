import argparse
import pathlib

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import tqdm
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder

from evaluation import compute_embedding
from models.dinov2.utils import DataAugmentation, Head, Loss, MultiCropWrapper, clip_gradients


def main():
    parser = argparse.ArgumentParser(
        "DINOv2 Training with ViT-g/14",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-b", "--batch-size", type=int, default=16)  # Reduced for ViT-g
    parser.add_argument("-d", "--device", type=str, choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("-l", "--logging-freq", type=int, default=200)
    parser.add_argument("--momentum-teacher", type=float, default=0.9995)
    parser.add_argument("-c", "--n-crops", type=int, default=4)
    parser.add_argument("-e", "--n-epochs", type=int, default=20)
    parser.add_argument("-o", "--out-dim", type=int, default=256)  # Increased from 10
    parser.add_argument("-t", "--tensorboard-dir", type=str, default="logs")
    parser.add_argument("--clip-grad", type=float, default=3.0)  # Increased for stability
    parser.add_argument("--norm-last-layer", action="store_true")
    parser.add_argument("--batch-size-eval", type=int, default=32)
    parser.add_argument("--teacher-temp", type=float, default=0.07)  # Adjusted for ViT-g
    parser.add_argument("--student-temp", type=float, default=0.1)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("-w", "--weight-decay", type=float, default=0.4)
    parser.add_argument("--early-stop-patience", type=int, default=5)

    args = parser.parse_args()
    print(vars(args))

    # Configuration for ViT-g/14
    MODEL_CONFIG = {
        "repo": "facebookresearch/dinov2",
        "name": "dinov2_vitg14",
        "dim": 1536,
        "patch_size": 14,
        "img_size": 224
    }

    vit_name, dim = "dinov2_vitg14", 384
    base_path = pathlib.Path("/home/alexsandro/pgcc/data/mestrado_Alexsandro/cross_validation/fsl/0_Amiloidose/fold0")

    path_dataset_train = base_path / "train"
    path_dataset_val = base_path / "val"

    logging_path = pathlib.Path(args.tensorboard_dir)
    device = torch.device(args.device)
    n_workers = 4

    # Data preparation (adjusted normalization for DINOv2)
    transform_aug = DataAugmentation(
        size=MODEL_CONFIG["img_size"],
        n_local_crops=args.n_crops - 2,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
    
    transform_plain = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((MODEL_CONFIG["img_size"], MODEL_CONFIG["img_size"])),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    # Dataset and DataLoader setup
    dataset_train_aug = ImageFolder(path_dataset_train, transform=transform_aug)
    dataset_train_plain = ImageFolder(path_dataset_train, transform=transform_plain)
    dataset_val_plain = ImageFolder(path_dataset_val, transform=transform_plain)

    data_loader_train_aug = DataLoader(
        dataset_train_aug,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=n_workers,
        pin_memory=True,
    )

    # Model initialization
    student_backbone = torch.hub.load(
        MODEL_CONFIG["repo"], 
        MODEL_CONFIG["name"], 
        pretrained=args.pretrained
    )
    teacher_backbone = torch.hub.load(
        MODEL_CONFIG["repo"], 
        MODEL_CONFIG["name"], 
        pretrained=args.pretrained
    )

    # DINO components
    student = MultiCropWrapper(
        student_backbone,
        Head(
            MODEL_CONFIG["dim"],
            args.out_dim,
            norm_last_layer=args.norm_last_layer,
            hidden_dim=2048  # Increased hidden dimension for ViT-g
        )
    ).to(device)
    
    teacher = MultiCropWrapper(
        teacher_backbone,
        Head(MODEL_CONFIG["dim"], args.out_dim)
    ).to(device)
    
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    # Loss and optimizer
    loss_inst = Loss(
        args.out_dim,
        teacher_temp=args.teacher_temp,
        student_temp=args.student_temp,
    ).to(device)

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=0.0005 * args.batch_size / 256,
        weight_decay=args.weight_decay
    )

    # Training loop
    n_batches = len(data_loader_train_aug)
    best_f1 = 0
    steps_since_best = 0
    n_steps = 0

    for epoch in range(args.n_epochs):
        for batch_idx, (images, _) in tqdm.tqdm(
            enumerate(data_loader_train_aug), total=n_batches
        ):
            if n_steps % args.logging_freq == 0:
                student.eval()
                
                # Compute embeddings and F1 score
                train_embs, _, train_labels = compute_embedding(
                    student.backbone, 
                    DataLoader(
                        dataset_train_plain,
                        batch_size=args.batch_size_eval,
                        num_workers=n_workers
                    ),
                    device=device
                )
                
                val_embs, _, val_labels = compute_embedding(
                    student.backbone,
                    DataLoader(
                        dataset_val_plain,
                        batch_size=args.batch_size_eval,
                        num_workers=n_workers
                    ),
                    device=device
                )
                
                # KNN classification
                knn = KNeighborsClassifier(n_neighbors=20)  # Increased neighbors for stability
                knn.fit(train_embs.cpu().numpy(), train_labels.cpu().numpy())
                val_pred = knn.predict(val_embs.cpu().numpy())
                current_f1 = f1_score(val_labels.cpu().numpy(), val_pred, pos_label=1)

                # Early stopping and logging
                if current_f1 > best_f1:
                    torch.save(student, logging_path / "best_model.pth")
                    best_f1 = current_f1
                    steps_since_best = 0
                else:
                    steps_since_best += 1
                    if steps_since_best >= args.early_stop_patience:
                        print(f"Early stopping at step {n_steps} | Best F1: {best_f1:.4f}")
                        return

                student.train()

            # Forward pass and optimization
            images = [img.to(device) for img in images]
            teacher_output = teacher(images[:2])
            student_output = student(images)
            loss = loss_inst(student_output, teacher_output)

            optimizer.zero_grad()
            loss.backward()
            clip_gradients(student, args.clip_grad)
            optimizer.step()

            # EMA update for teacher
            with torch.no_grad():
                for s_param, t_param in zip(student.parameters(), teacher.parameters()):
                    t_param.data.mul_(args.momentum_teacher)
                    t_param.data.add_((1 - args.momentum_teacher) * s_param.detach().data)

            n_steps += 1


if __name__ == "__main__":
    main()