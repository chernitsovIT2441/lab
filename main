"""
Проект: Классификация состояния растений по мультиспектральным снимкам
Модель: EfficientNet-B0 (CNN)
Датасет: PlantVillage
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm
import warnings
import random
from collections import Counter
import pandas as pd
warnings.filterwarnings('ignore')

# ==================== КОНФИГУРАЦИЯ ====================
class Config:
    # Пути к данным (ИЗМЕНИТЕ НА ВАШ ПУТЬ)
    data_path = './PlantVillage-Dataset/raw/color'
    
    # Параметры модели
    model_name = 'efficientnet'  # resnet18, resnet50, efficientnet, vit
    num_classes = 38
    batch_size = 32
    num_epochs = 20
    learning_rate = 0.001
    image_size = 224
    
    # Параметры обучения
    use_weighted_loss = True
    save_dir = './plant_classification_results'
    model_save_path = './best_plant_model.pth'

# Создание директорий
os.makedirs(Config.save_dir, exist_ok=True)

# Установка устройства
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используется устройство: {device}")

# Установка seed для воспроизводимости
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ==================== КЛАССЫ ДЛЯ ЗАГРУЗКИ ДАННЫХ ====================
class PlantVillageDataset(Dataset):
    """Кастомный датасет для PlantVillage"""
    
    def __init__(self, root_dir, transform=None, is_train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        
        # Проверка существования директории
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Директория {root_dir} не найдена!")
        
        # Собираем классы
        classes = sorted([d for d in os.listdir(root_dir) 
                         if os.path.isdir(os.path.join(root_dir, d))])
        
        if len(classes) == 0:
            raise ValueError(f"В директории {root_dir} не найдены классы!")
        
        print(f"Найдено классов: {len(classes)}")
        
        for idx, class_name in enumerate(classes):
            self.class_to_idx[class_name] = idx
            class_dir = os.path.join(root_dir, class_name)
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(idx)
        
        print(f"Всего изображений: {len(self.images)}")
        
        # Выводим распределение классов
        label_counts = Counter(self.labels)
        print("\nРаспределение классов (первые 5):")
        for i, (class_name, idx) in enumerate(self.class_to_idx.items()):
            if i < 5:
                print(f"{class_name}: {label_counts[idx]}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Ошибка при загрузке {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.images))

# ==================== ФУНКЦИИ ПРЕДОБРАБОТКИ ====================
def get_transforms(is_train=True):
    """Получение преобразований для изображений"""
    
    if is_train:
        return transforms.Compose([
            transforms.Resize((Config.image_size + 32, Config.image_size + 32)),
            transforms.RandomResizedCrop(Config.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((Config.image_size, Config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def calculate_class_weights(dataset):
    """Вычисление весов классов для борьбы с дисбалансом"""
    labels = dataset.labels
    class_counts = Counter(labels)
    
    total_samples = len(labels)
    num_classes = len(class_counts)
    
    weights = []
    for i in range(num_classes):
        weight = total_samples / (num_classes * class_counts[i])
        weights.append(weight)
    
    return torch.FloatTensor(weights)

def prepare_data():
    """Подготовка данных для обучения"""
    
    print("Загрузка данных...")
    
    # Создание датасета
    full_dataset = PlantVillageDataset(
        root_dir=Config.data_path,
        transform=get_transforms(is_train=True)
    )
    
    # Вычисление весов классов
    class_weights = calculate_class_weights(full_dataset)
    
    # Разделение на обучающую, валидационную и тестовую выборки
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Применение соответствующих преобразований
    train_dataset.dataset.transform = get_transforms(is_train=True)
    val_dataset.dataset.transform = get_transforms(is_train=False)
    test_dataset.dataset.transform = get_transforms(is_train=False)
    
    # Создание DataLoader'ов
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.batch_size, 
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=Config.batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"\nРазмер обучающей выборки: {len(train_dataset)}")
    print(f"Размер валидационной выборки: {len(val_dataset)}")
    print(f"Размер тестовой выборки: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, class_weights, full_dataset.class_to_idx

# ==================== СОЗДАНИЕ МОДЕЛИ ====================
def create_model(num_classes, model_name='efficientnet'):
    """Создание модели CNN"""
    
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    elif model_name == 'efficientnet':
        model = models.efficientnet_b0(pretrained=True)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    elif model_name == 'vit':
        model = models.vit_b_16(pretrained=True)
        num_features = model.heads.head.in_features
        model.heads.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    else:
        raise ValueError(f"Неизвестная модель: {model_name}")
    
    return model

# ==================== ФУНКЦИИ ОБУЧЕНИЯ ====================
def train_epoch(model, loader, criterion, optimizer, device):
    """Обучение на одну эпоху"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc='Обучение')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return epoch_loss, epoch_acc, epoch_f1

def validate_epoch(model, loader, criterion, device):
    """Валидация модели"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Валидация'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return epoch_loss, epoch_acc, epoch_f1, all_preds, all_labels

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
    """Полный цикл обучения"""
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_f1s, val_f1s = [], []
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        print(f'\nЭпоха {epoch+1}/{num_epochs}')
        print('-' * 50)
        
        # Обучение
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_f1s.append(train_f1)
        
        # Валидация
        val_loss, val_acc, val_f1, _, _ = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)
        
        # Сохранение лучшей модели
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            torch.save(model.state_dict(), Config.model_save_path)
            print(f"✓ Сохранена лучшая модель с accuracy: {val_acc:.4f}")
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')
    
    # Загрузка лучшей модели
    model.load_state_dict(best_model_state)
    
    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'train_f1s': train_f1s,
        'val_f1s': val_f1s
    }

# ==================== ФУНКЦИИ ВИЗУАЛИЗАЦИИ ====================
def plot_training_history(history):
    """Визуализация истории обучения"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # График потерь
    axes[0].plot(history['train_losses'], label='Train')
    axes[0].plot(history['val_losses'], label='Validation')
    axes[0].set_xlabel('Эпоха')
    axes[0].set_ylabel('Потери')
    axes[0].set_title('График потерь')
    axes[0].legend()
    axes[0].grid(True)
    
    # График точности
    axes[1].plot(history['train_accs'], label='Train')
    axes[1].plot(history['val_accs'], label='Validation')
    axes[1].set_xlabel('Эпоха')
    axes[1].set_ylabel('Точность')
    axes[1].set_title('График точности')
    axes[1].legend()
    axes[1].grid(True)
    
    # График F1-score
    axes[2].plot(history['train_f1s'], label='Train')
    axes[2].plot(history['val_f1s'], label='Validation')
    axes[2].set_xlabel('Эпоха')
    axes[2].set_ylabel('F1-score')
    axes[2].set_title('График F1-score')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.save_dir, 'training_history.png'))
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Построение матрицы ошибок"""
    
    # Используем только первые 10 классов для читаемости
    if len(class_names) > 10:
        class_names = list(class_names.keys())[:10]
        # Фильтруем предсказания
        mask = y_true < 10
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    plt.title('Матрица ошибок (первые 10 классов)')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.save_dir, 'confusion_matrix.png'))
    plt.show()

def visualize_predictions(model, test_loader, class_names, device, num_samples=16):
    """Визуализация предсказаний"""
    
    model.eval()
    
    # Получаем батч изображений
    images, labels = next(iter(test_loader))
    images, labels = images[:num_samples].to(device), labels[:num_samples]
    
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    
    # Визуализация
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.ravel()
    
    class_names_list = list(class_names.keys())
    
    for i in range(num_samples):
        img = images[i].cpu().numpy().transpose((1, 2, 0))
        
        # Денормализация
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].axis('off')
        
        true_label = class_names_list[labels[i]]
        pred_label = class_names_list[preds[i]]
        
        color = 'green' if labels[i] == preds[i] else 'red'
        axes[i].set_title(f'Истина: {true_label[:20]}\nПредск: {pred_label[:20]}', 
                         color=color, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.save_dir, 'predictions_visualization.png'))
    plt.show()

def analyze_class_performance(test_labels, test_preds, class_names):
    """Анализ производительности по классам"""
    
    from sklearn.metrics import precision_recall_fscore_support
    
    class_names_list = list(class_names.keys())
    
    precision, recall, f1, support = precision_recall_fscore_support(
        test_labels, test_preds, average=None
    )
    
    # Создаем DataFrame
    results_df = pd.DataFrame({
        'Класс': class_names_list[:len(precision)],
        'Точность': precision,
        'Полнота': recall,
        'F1-score': f1,
        'Поддержка': support
    })
    
    results_df = results_df.sort_values('F1-score', ascending=False)
    
    print("\n" + "=" * 80)
    print("ДЕТАЛЬНЫЙ АНАЛИЗ ПО КЛАССАМ (первые 10)")
    print("=" * 80)
    print(results_df.head(10).to_string(index=False))
    
    return results_df

# ==================== ОСНОВНАЯ ФУНКЦИЯ ====================
def main():
    """Основная функция"""
    
    print("=" * 60)
    print("КЛАССИФИКАЦИЯ СОСТОЯНИЯ РАСТЕНИЙ")
    print("=" * 60)
    
    try:
        # Подготовка данных
        print("\n1. ПОДГОТОВКА ДАННЫХ")
        print("-" * 40)
        train_loader, val_loader, test_loader, class_weights, class_to_idx = prepare_data()
        
        # Создание модели
        print("\n2. СОЗДАНИЕ МОДЕЛИ")
        print("-" * 40)
        model = create_model(Config.num_classes, model_name=Config.model_name)
        model = model.to(device)
        
        # Информация о модели
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Всего параметров: {total_params:,}")
        print(f"Обучаемых параметров: {trainable_params:,}")
        
        # Настройка обучения
        print("\n3. НАСТРОЙКА ОБУЧЕНИЯ")
        print("-" * 40)
        
        if Config.use_weighted_loss:
            class_weights = class_weights.to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            print("Используется взвешенная функция потерь")
        else:
            criterion = nn.CrossEntropyLoss()
        
        optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
        
        # Обучение
        print("\n4. ОБУЧЕНИЕ МОДЕЛИ")
        print("-" * 40)
        model, history = train_model(
            model, train_loader, val_loader, criterion, optimizer, 
            device, Config.num_epochs
        )
        
        # Визуализация
        print("\n5. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
        print("-" * 40)
        plot_training_history(history)
        
        # Оценка на тесте
        print("\n6. ОЦЕНКА МОДЕЛИ")
        print("-" * 40)
        
        test_loss, test_acc, test_f1, test_preds, test_labels = validate_epoch(
            model, test_loader, criterion, device
        )
        
        print(f"\nРезультаты на тестовой выборке:")
        print(f"Точность (Accuracy): {test_acc:.4f}")
        print(f"F1-score: {test_f1:.4f}")
        print(f"Потери: {test_loss:.4f}")
        
        # Матрица ошибок
        plot_confusion_matrix(test_labels, test_preds, class_to_idx)
        
        # Визуализация предсказаний
        visualize_predictions(model, test_loader, class_to_idx, device)
        
        # Анализ по классам
        analyze_class_performance(test_labels, test_preds, class_to_idx)
        
        # Сохранение результатов
        print("\n7. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
        print("-" * 40)
        
        with open(os.path.join(Config.save_dir, 'results.txt'), 'w', encoding='utf-8') as f:
            f.write("РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Модель: {Config.model_name}\n")
            f.write(f"Тестовая точность: {test_acc:.4f}\n")
            f.write(f"Тестовый F1-score: {test_f1:.4f}\n")
        
        print(f"✓ Результаты сохранены в: {Config.save_dir}")
        print(f"✓ Модель сохранена в: {Config.model_save_path}")
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        print("\nПроверьте:")
        print("1. Правильно ли указан путь к данным:", Config.data_path)
        print("2. Установлены ли все зависимости")
        print("3. Достаточно ли места на диске")

# ==================== ЗАПУСК ====================
if __name__ == "__main__":
    main()
