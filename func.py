import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Sampler
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms
import numpy as np
from sklearn import metrics

def load_data(datasets_dir: str, batch_size: int):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = CIFAR10(root=datasets_dir, train=True,
                        download=True, transform=transform)
    torch.manual_seed(43)
    val_size = int(len(train_set)*0.1)
    train_size = len(train_set) - val_size
    train_ds, val_ds = torch.utils.data.random_split(train_set,
                                                     [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=2)

    val_loader = DataLoader(val_ds, batch_size=batch_size*2,
                            shuffle=True, num_workers=2)

    test_set = CIFAR10(root=datasets_dir, train=False,
                       download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size*2,
                             shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, train_set.classes

def get_subset(dataset, used_classes, new_labels):
    new_data = []
    new_targets = []
    for img, label in zip(dataset.data, dataset.targets):
        class_name = dataset.classes[label]
        if class_name in used_classes:
            new_data.append(img)
            new_target = new_labels[used_classes.index(class_name)]
            new_targets.append(new_target)
    new_data = np.array(new_data)
    new_targets = np.array(new_targets)
    dataset.data = new_data
    dataset.targets = new_targets

    return dataset


def load_superclass(datasets_dir: str, batch_size: int):
    used_classes = ['bear', 'leopard', 'lion', 'tiger', 'wolf']
    new_labels = [10, 11, 12, 13, 14]

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = CIFAR100(root=datasets_dir, train=True,
                         download=True, transform=transform)
    train_set = get_subset(train_set, used_classes, new_labels)

    torch.manual_seed(43)
    val_size = int(len(train_set)*0.1)
    train_size = len(train_set) - val_size
    train_ds, val_ds = torch.utils.data.random_split(train_set,
                                                     [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=2)

    val_loader = DataLoader(val_ds, batch_size=batch_size*2,
                            shuffle=True, num_workers=2)

    test_set = CIFAR100(root=datasets_dir, train=False,
                       download=True, transform=transform)
    test_set = get_subset(test_set, used_classes, new_labels)
    test_loader = DataLoader(test_set, batch_size=batch_size*2,
                             shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, used_classes

def accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    return torch.tensor(torch.sum(predicted == labels).item() / len(predicted))

def train(net, train_loader, val_loader, filename: str, device, **params):
    print("="*50)
    print("Training...")

    num_epochs = params['num_epochs']
    lr = params['learning_rate']
    check_limit = params['check_limit']

    losses = []
    accs = []

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr)

    n_steps = len(train_loader)
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        # training
        net.train()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (i+1) % check_limit == 0:
                l = running_loss / check_limit
                running_loss = 0.0

                acc = correct / total
                correct = 0
                total = 0

                print(f"Epoch [{epoch+1}/{num_epochs}], " \
                      f"Step [{i+1}/{n_steps}], " \
                      f"Loss: {l:.4f}, " \
                      f"Accuracy: {acc:.4f}")

        # validation
        val_losses = []
        val_accs = []
        net.eval()
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                acc = accuracy(outputs, labels)

                val_losses.append(loss)
                val_accs.append(acc)

        epoch_loss = torch.stack(val_losses).mean().item()
        epoch_acc = torch.stack(val_accs).mean().item()

        losses.append(epoch_loss)
        accs.append(epoch_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}], " \
              f"val_loss: {epoch_loss:.4f}, " \
              f"val_accuracy: {epoch_acc:.4f}")

        torch.save(net.state_dict(), f"weights/Epoch_{epoch+1}_ConvNet.pth")

    print("Finished")

    return losses, accs

def evaluate(net, data_loader, classes, device, tag: str = ""):
    print("="*50)
    print(tag)
    print("Evaluating...")

    num_classes = len(classes)

    total = 0
    correct = 0
    class_correct = list(0 for i in range(num_classes))
    class_total = list(0 for i in range(num_classes))

    y_true = np.array([])
    y_pred = np.array([])

    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data
            y_true = np.append(y_true, labels.numpy())
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            y_pred = np.append(y_pred, predicted.detach().cpu().numpy())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    accuracy = correct / total
    print(f"Overall accuracy: {accuracy:.4f}")

    precision = metrics.precision_score(y_true, y_pred, average="weighted")
    recall = metrics.recall_score(y_true, y_pred, average="weighted")
    f1 = metrics.f1_score(y_true, y_pred, average="weighted")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 score: {f1:.4f}")

    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    print(f"Confusion matrix:\n{confusion_matrix}")

    print("-"*50)
    print("Accuracy by classes")
    for i in range(num_classes):
        if class_total[i] == 0:
            continue
        acc = class_correct[i] / class_total[i]
        print(f"{classes[i]}: {acc:.4f}")

    print("\nFinished")

    return confusion_matrix

