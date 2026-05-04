import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import Food101
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from datasets import load_dataset
import os
import warnings

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

class AlexNet(nn.Module):
    def __init__(self, num_classes=101):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class CIFAR10HF(Dataset):
    def __init__(self, hf_dataset, split, transform=None):
        self.data = hf_dataset[split]
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img = self.data[idx]['img']
        label = self.data[idx]['label']
        if self.transform:
            img = self.transform(img)
        return img, label

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("-------------------------------")
    print(f"SYSTEM READY | Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("-------------------------------")

    cifar_train_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])
    cifar_test_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])

    cifar_hf = load_dataset("cifar10")
    cifar_train = CIFAR10HF(cifar_hf, 'train', transform=cifar_train_transforms)
    cifar_test  = CIFAR10HF(cifar_hf, 'test',  transform=cifar_test_transforms)
    cifar_train_loader = DataLoader(cifar_train, batch_size=64, shuffle=True)
    cifar_test_loader  = DataLoader(cifar_test,  batch_size=64, shuffle=False)

    print("-------------------------------")
    print("ALEXNET FROM SCRATCH (CIFAR-10)...")
    print("-------------------------------")
    model_cifar = AlexNet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_cifar = optim.Adam(model_cifar.parameters(), lr=0.001)

    cifar_acc_history = []
    EPOCHS_CIFAR = 10
    for epoch in range(EPOCHS_CIFAR):
        _, t_acc = train(model_cifar, cifar_train_loader, optimizer_cifar, criterion, device)
        _, v_acc = evaluate(model_cifar, cifar_test_loader, criterion, device)
        cifar_acc_history.append(v_acc)
        print(f"CIFAR10 - Epoch {epoch+1:02d} | Train Accuracy: {t_acc:.4f} | Test Accuracy: {v_acc:.4f}")

    food_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    food_train_dataset = Food101(root='./data', split='train', transform=food_transforms, download=True)
    food_test_dataset  = Food101(root='./data', split='test',  transform=food_transforms, download=False)
    food_train_loader = DataLoader(food_train_dataset, batch_size=64, shuffle=True)
    food_test_loader = DataLoader(food_test_dataset, batch_size=64, shuffle=False)

    print("-------------------------------")
    print("\nALEXNET FROM SCRATCH (FOOD-101)...")
    print("-------------------------------")
    model_food_scratch = AlexNet(num_classes=101).to(device)
    optimizer_scratch = optim.Adam(model_food_scratch.parameters(), lr=0.0001)
    
    scratch_history = []
    EPOCHS_FOOD = 10
    for epoch in range(EPOCHS_FOOD):
        _, t_acc = train(model_food_scratch, food_train_loader, optimizer_scratch, criterion, device)
        _, v_acc = evaluate(model_food_scratch, food_test_loader, criterion, device)
        scratch_history.append(v_acc)
        print(f"FOOD-SCRATCH - Epoch {epoch+1:02d} | Train Acc: {t_acc:.4f} | Test Acc: {v_acc:.4f}")

    print("-------------------------------")
    print("\nTRANSFER LEARNING VGG16 (FOOD-101)...")
    print("-------------------------------")
    vgg16 = models.vgg16(weights='IMAGENET1K_V1')
    for param in vgg16.parameters():
        param.requires_grad = False 
    
    vgg16.classifier[6] = nn.Linear(4096, 101)
    vgg16 = vgg16.to(device)
    optimizer_vgg = optim.Adam(vgg16.classifier[6].parameters(), lr=0.001)
    
    vgg_history = []
    for epoch in range(EPOCHS_FOOD):
        _, t_acc = train(vgg16, food_train_loader, optimizer_vgg, criterion, device)
        _, v_acc = evaluate(vgg16, food_test_loader, criterion, device)
        vgg_history.append(v_acc)
        print(f"FOOD-VGG16 - Epoch {epoch+1:02d} | Train Accuracy: {t_acc:.4f} | Test Accuracy: {v_acc:.4f}")

    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(range(1, len(cifar_acc_history)+1), cifar_acc_history, marker='o', color='green')
    plt.title('AlexNet Accuracy (CIFAR-10)')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(range(1, EPOCHS_FOOD+1), scratch_history, label='From Scratch', marker='s')
    plt.plot(range(1, EPOCHS_FOOD+1), vgg_history, label='Transfer Learning', marker='d')
    plt.title('Food-101: Scratch vs TL')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(range(1, 11), cifar_acc_history[:10], label='AlexNet (CIFAR-10)', color='green', linestyle='--')
    plt.plot(range(1, 11), scratch_history, label='AlexNet (Food-101)', color='blue')
    plt.title('Complexity Comparison')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.show()
