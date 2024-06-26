import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import Swinv2Config, Swinv2ForImageClassification
from tqdm import tqdm
import time
import logging
import os
import random
import matplotlib.pyplot as plt
import numpy as np


def configure_logging():
    os.makedirs('/root/swin/results', exist_ok=True)
    os.makedirs('/root/swin/results/vis', exist_ok=True)
    logging.basicConfig(filename='/root/swin/results/training_log.log', level=logging.INFO,
                        format='%(asctime)s %(message)s')


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transforms():
    cifar100_mean = [0.5071, 0.4867, 0.4408]
    cifar100_std = [0.2675, 0.2565, 0.2761]
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std),
    ])


def get_data_loaders(transform):
    train_dataset = datasets.CIFAR100(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR100(root='./data', train=False, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
    return train_loader, test_loader


def initialize_model(device):
    config = Swinv2Config(num_labels=100)  # CIFAR-100有100个类别
    model = Swinv2ForImageClassification(config)
    model = model.to(device)
    return model


def load_checkpoint(model, optimizer, checkpoint_path):
    start_epoch = 0
    best_acc1 = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc1 = checkpoint['best_acc1']
        logging.info(f'Loaded checkpoint from epoch {start_epoch}')
    return start_epoch, best_acc1


def visualize_results(images, labels, predictions, epoch, cifar100_mean, cifar100_std):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(10):
        image = images[i].cpu().numpy().transpose((1, 2, 0))
        mean = np.array(cifar100_mean)
        std = np.array(cifar100_std)
        image = std * image + mean
        image = np.clip(image, 0, 1)
        ax = axes[i // 5, i % 5]
        ax.imshow(image)
        ax.set_title(f'True: {labels[i]}, Pred: {predictions[i]}')
        ax.axis('off')
    plt.savefig(f'/root/swin/results/vis/epoch_{epoch}.png')
    plt.close()


def evaluate_model(model, test_loader, criterion, epoch, device, cifar100_mean, cifar100_std):
    model.eval()
    correct = 0
    total = 0
    running_test_loss = 0.0
    all_images, all_labels, all_predictions = [], [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            running_test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 收集所有图片、标签和预测值
            all_images.extend(images.cpu())
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    acc1 = 100 * correct / total
    avg_test_loss = running_test_loss / len(test_loader)
    logging.info(f'Test Loss: {avg_test_loss:.4f}, ACC@1: {acc1:.2f}%')
    print(f'ACC@1 of the model on the CIFAR-100 test images: {acc1:.2f}%')

    # 随机选择10张图片进行可视化
    if len(all_images) > 0:
        indices = random.sample(range(len(all_images)), 10)
        selected_images = [all_images[i] for i in indices]
        selected_labels = [all_labels[i] for i in indices]
        selected_predictions = [all_predictions[i] for i in indices]
        visualize_results(selected_images, selected_labels, selected_predictions, epoch, cifar100_mean, cifar100_std)

    return acc1


def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, eval_interval, start_epoch, device,
                cifar100_mean, cifar100_std):
    global best_acc1
    logging.info(f'Starting training at: {time.time()}')

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        # 创建一个进度条
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{num_epochs}', unit='batch') as pbar:
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images).logits
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
                pbar.update(1)

        epoch_time = time.time() - start_time
        remaining_time = epoch_time * (num_epochs - epoch - 1)
        avg_loss = running_loss / len(train_loader)
        logging.info(
            f'Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s, Remaining time: {remaining_time / 60:.2f}min')
        print(
            f"Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s, Remaining time: {remaining_time / 60:.2f}min")

        # 每 eval_interval 个 epoch 进行一次测试
        if (epoch) % eval_interval == 0 or (epoch) == num_epochs:
            print("Testing......")
            acc1 = evaluate_model(model, test_loader, criterion, epoch, device, cifar100_mean, cifar100_std)
            # 保存最优模型
            if acc1 > best_acc1:
                best_acc1 = acc1
                model.save_pretrained('/root/swin/results/best_model')
                logging.info(f'Saved best model with ACC@1: {acc1:.2f}%')

        # 保存checkpoint
        checkpoint_path = '/root/swin/results/checkpoint.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc1': best_acc1
        }, checkpoint_path)
        logging.info(f'Saved checkpoint at epoch {epoch}')


def main():
    configure_logging()
    device = get_device()
    transform = get_transforms()
    train_loader, test_loader = get_data_loaders(transform)
    model = initialize_model(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    checkpoint_path = '/root/swin/results/checkpoint.pth'
    start_epoch, best_acc1 = load_checkpoint(model, optimizer, checkpoint_path)
    train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=1000, eval_interval=1,
                start_epoch=start_epoch, device=device, cifar100_mean=[0.5071, 0.4867, 0.4408],
                cifar100_std=[0.2675, 0.2565, 0.2761])


if __name__ == "__main__":
    main()
