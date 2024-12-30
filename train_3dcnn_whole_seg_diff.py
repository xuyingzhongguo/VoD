import torch
import argparse
import torch.nn as nn
from torchvision.transforms import v2, InterpolationMode
from torch.utils.data import DataLoader
from datasets.FFpp import FfppDataset_temporal_whole_diff
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix
import os
import sys

# Set the random seed for all devices (CPU and all GPUs)
torch.manual_seed(42)

# If you are using CUDA (GPU), set this as well for ensuring reproducibility on GPU operations
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def main(args):
    epochs = args.epoch
    learning_rate = args.lr
    batch_size = args.bs
    image_size = args.imgsz
    save_path = args.save_path

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    model_save_path = os.path.join(save_path, f'{args.ds}', f'{args.model_name}_{args.train_frames}_{args.seg_len}_{args.steps}_{args.interval}')
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    log_save_path = os.path.join(args.log_path, f'{args.ds}', f'{args.model_name}_{args.train_frames}_{args.seg_len}_{args.steps}_{args.interval}')
    if not os.path.exists(log_save_path):
        os.makedirs(log_save_path)

    file = open(os.path.join(log_save_path, f"train.txt"), "a")
    sys.stdout = file
    print("---" * 30)
    for arg in vars(args):
        num_space = 25 - len(arg)
        print(arg + " " * num_space + str(getattr(args, arg)))
    print("---" * 30)

    ### Data ###
    real_transform = v2.Compose([
        v2.Resize(size=(int(image_size*1.2), int(image_size*1.2)), antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        v2.CenterCrop((image_size, image_size))
    ])

    custom_frames = {'test': args.test_frames, 'train': args.train_frames, 'val': args.val_frames}
    train_dataset = FfppDataset_temporal_whole_diff(mode="train", dataset=args.ds,
                                                    transform=real_transform,
                                                    segment_len=args.seg_len,
                                                    square_scale=args.square_scale,
                                                    frames=custom_frames,
                                                    step=args.steps,
                                                    interval=args.interval,
                                                    quality=args.quality)
    val_dataset = FfppDataset_temporal_whole_diff(mode="val", dataset=args.ds,
                                                  transform=real_transform,
                                                  segment_len=args.seg_len,
                                                  square_scale=args.square_scale,
                                                  frames=custom_frames,
                                                  step=args.steps,
                                                  interval=args.interval,
                                                  quality=args.quality)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
    print(f'{len(train_dataset)} Data loaded!')

    ### Model ###
    if args.model_name == 'slow_r50':
        # (2, 3, >10, 224, 224)
        model = torch.hub.load("facebookresearch/pytorchvideo:main", model='slow_r50', pretrained=False)
        model.blocks[5].proj = nn.Linear(in_features=2048, out_features=2, bias=True)
    elif args.model_name == 'x3d_s':
        # (2, 3, >13, 224, 224)
        model = torch.hub.load("facebookresearch/pytorchvideo:main", model='x3d_s', pretrained=False)
        model.blocks[5].proj = nn.Linear(in_features=2048, out_features=2, bias=True)
    elif args.model_name == 'x3d_m':
        # (2, 3, >13, 224, 224)
        model = torch.hub.load("facebookresearch/pytorchvideo:main", model='x3d_m', pretrained=False)
        model.blocks[5].proj = nn.Linear(in_features=2048, out_features=2, bias=True)
    elif args.model_name == 'efficient_x3d_s':
        # (2, 3, -, 224, 224)
        model = torch.hub.load("facebookresearch/pytorchvideo:main", model='efficient_x3d_s', pretrained=False)
        model.projection = nn.Linear(in_features=2048, out_features=2, bias=True)
    elif args.model_name == 'mvit_base_16x4':
        # (2, 3, 16, 224, 224)
        model = torch.hub.load("facebookresearch/pytorchvideo:main", model='mvit_base_16x4', pretrained=False)
        model.head.proj = nn.Linear(in_features=768, out_features=2, bias=True)
    elif args.model_name == 'mvit_base_32x3':
        # (2, 3, 32, 224, 224)
        model = torch.hub.load("facebookresearch/pytorchvideo:main", model='mvit_base_32x3', pretrained=False)
        model.head.proj = nn.Linear(in_features=768, out_features=2, bias=True)
    else:
        raise "Wrong model name"

    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40], gamma=0.1)
    best_val_auc = 0

    ### Train ###
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_labels = []
        train_score = []
        train_preds = []

        for segment, label, _, _, _ in train_loader:
            segment, label = segment.to(device), label.to(device)
            output = model(segment)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_score.extend(output[:, 1].detach().cpu().numpy().flatten())
            _, predicted = torch.max(output.data, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(label.cpu().numpy().flatten())

            # Calculate metrics
        train_auc = roc_auc_score(train_labels, train_score)
        train_accuracy = balanced_accuracy_score(train_labels, train_preds)

        print(
            f'Epoch {epoch + 1}: Train Loss: {train_loss / len(train_loader):.4f}, AUC: {train_auc:.4f}, Accuracy: {train_accuracy:.4f}')
        con_mat = confusion_matrix(train_labels, train_preds, labels=[0, 1])
        print(con_mat)

        ### Validation ###
        model.eval()
        val_loss = 0
        val_labels = []
        val_score = []
        val_preds = []
        with torch.no_grad():
            for segment, label, _, _, _ in val_loader:
                segment, label = segment.to(device), label.to(device)
                output = model(segment)
                loss = criterion(output, label)
                val_loss += loss.item()

                val_score.extend(output[:, 1].cpu().numpy().flatten())
                _, predicted = torch.max(output.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(label.cpu().numpy().flatten())

        val_auc = roc_auc_score(val_labels, val_score)
        val_accuracy = balanced_accuracy_score(val_labels, val_preds)

        print(f'Validation Loss: {val_loss / len(val_loader):.4f}, AUC: {val_auc:.4f}, Accuracy: {val_accuracy:.4f}')
        con_mat = confusion_matrix(val_labels, val_preds, labels=[0, 1])
        print(con_mat)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), os.path.join(model_save_path, 'best.pth'))
            print(f'New best validation auc: {best_val_auc:.4f}, saved model.')

        print('==========================================================')
        scheduler.step()
        print(scheduler.get_last_lr())

    sys.stdout = sys.__stdout__
    file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds', nargs='+', help='List of datasets', default=[])
    parser.add_argument('--model_name', type=str, default='slow_r50')
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--imgsz', type=int, default=256)
    parser.add_argument('--train_frames', type=int, default=15)
    parser.add_argument('--val_frames', type=int, default=5)
    parser.add_argument('--test_frames', type=int, default=5)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--interval', type=int, default=1)
    parser.add_argument('--seg_len', type=int, default=10)
    parser.add_argument('--square_scale', type=float, default=1.2)
    parser.add_argument('--quality', type=str, default='c23')
    parser.add_argument('--save_path', type=str, default='./checkpoints')
    parser.add_argument('--log_path', type=str, default='./logs')
    args = parser.parse_args()
    main(args)
