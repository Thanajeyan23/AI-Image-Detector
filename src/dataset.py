from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloaders(data_dir, batch_size=64):
    # ImageNet normalization — required for pretrained ResNet18
    # Without this, the model trains very poorly
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    # Training transforms: augmentation helps the model generalize better
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),          # Random mirror
        transforms.RandomRotation(10),              # Small random rotation
        transforms.ColorJitter(brightness=0.2,      # Slight color changes
                               contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    # Validation transforms: no augmentation, just resize + normalize
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    train_dataset = datasets.ImageFolder(root=f'{data_dir}/train', transform=train_transform)
    val_dataset   = datasets.ImageFolder(root=f'{data_dir}/val',   transform=val_transform)

    # num_workers=4 uses 4 CPU cores to load data in parallel — much faster
    # num_workers=0 required on Windows (no fork support)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)

    print(f"Train: {len(train_dataset)} images | Val: {len(val_dataset)} images")
    print(f"Classes: {train_dataset.classes}")  # ['fake', 'real']

    return train_loader, val_loader