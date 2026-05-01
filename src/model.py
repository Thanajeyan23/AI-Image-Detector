from torchvision import models
import torch.nn as nn


def get_model():
    """
    ResNet18 pre-trained on ImageNet, fine-tuned for binary classification.
    
    Transfer Learning explained:
    - ResNet18 was trained on 1.2 million ImageNet images
    - It already knows how to detect textures, edges, patterns
    - We just replace the last layer (fc) to output 2 classes: fake / real
    - This is much better than training from scratch
    """
    # weights='DEFAULT' is the modern way (pretrained=True is deprecated)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze early layers — they already learned general features
    # We only train the last few layers + our new classification head
    for name, param in model.named_parameters():
        if 'layer4' not in name and 'fc' not in name:
            param.requires_grad = False

    # Replace the final fully-connected layer
    # ResNet18's original fc: 512 -> 1000 (ImageNet classes)
    # Ours: 512 -> 2 (fake or real)
    num_features = model.fc.in_features  # 512
    model.fc = nn.Sequential(
        nn.Dropout(0.3),           # Dropout helps prevent overfitting
        nn.Linear(num_features, 2) # 2 output classes
    )

    return model