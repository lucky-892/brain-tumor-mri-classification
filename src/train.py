import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.optim as optim
import os
from torch import nn
import copy
import matplotlib.pyplot as plt
import numpy as np

# Параметры
batch_size = 32
epochs = 10
learning_rate = 0.001

# Преобразования для изображения
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Загрузка данных
train_data = datasets.ImageFolder(root='training', transform=transform)
test_data = datasets.ImageFolder(root='testing', transform=transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Загрузка модели ResNet18
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(in_features=512, out_features=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# --- НОВОЕ: Сохраняем начальные веса последнего слоя (ДО обучения) ---
# Мы берем именно слой fc, так как он создается с нуля случайными числами
initial_weights = copy.deepcopy(model.fc.weight.detach().cpu().numpy())

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# --- НОВОЕ: Списки для отслеживания метрик ---
history_loss = []
history_acc = []

print("Начинаем тренировку...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    # Сохраняем в историю
    history_loss.append(epoch_loss)
    history_acc.append(epoch_acc)
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

torch.save(model.state_dict(), 'resnet18_brain_tumor_model.pth')
print("Модель сохранена!")

# Оценка на тестовых данных
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# --- НОВОЕ: Сохраняем финальные веса (ПОСЛЕ обучения) ---
final_weights = model.fc.weight.detach().cpu().numpy()

# ==========================================
# БЛОК ВИЗУАЛИЗАЦИИ ДЛЯ ОТЧЕТА ПРЕПОДАВАТЕЛЮ
# ==========================================

# 1. График процесса обучения (Loss и Accuracy)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), history_loss, marker='o', color='red', label='Training Loss')
plt.title('Изменение ошибки (Loss) по эпохам')
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), history_acc, marker='o', color='green', label='Training Accuracy')
plt.title('Изменение точности (Accuracy) по эпохам')
plt.xlabel('Эпоха')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('training_curves.png') # Сохраняем картинку для отчета
plt.show()

# 2. Гистограмма распределения весов (До и После)
plt.figure(figsize=(10, 6))
# Сглаживаем массивы весов в одномерный список
plt.hist(initial_weights.flatten(), bins=50, alpha=0.5, color='blue', label='До обучения (Инициализация)')
plt.hist(final_weights.flatten(), bins=50, alpha=0.5, color='orange', label='После обучения (Оптимизированные)')

plt.title('Распределение весов (Weights) классификатора до и после обучения')
plt.xlabel('Значение веса')
plt.ylabel('Частота (Количество связей с таким весом)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('weights_histogram.png') # Сохраняем картинку для отчета
plt.show()