import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import os

from convnet import *
from func import *

if __name__ == '__main__':
    # params
    datasets_dir = '/datasets'
    net_type = 'ConvNet'
    train_model: bool = False
    evaluate_model: bool = True
    use_existing_weights: bool = True
    add_superclass: bool = True
    save_plots: bool = False

    print(f'Using {net_type}')

    # hyper-parameters
    params: dict = {
        "num_epochs": 20,
        "batch_size": 100,
        "learning_rate": 0.001,
        "check_limit": 50
    }

    if add_superclass:
        params: dict = {
            "num_epochs": 30,
            "batch_size": 100,
            "learning_rate": 0.0001,
            "check_limit": 10
        }

    # set cuda if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print("Hooray! Running on GPU!")

    # load data
    train_loader, val_loader, test_loader, cifar10_classes = load_data(datasets_dir,
                                                                       params['batch_size'])
    all_classes = cifar10_classes
    if add_superclass:
        train_loader, val_loader, test_loader, add_classes = load_superclass(datasets_dir,
                                                                             params['batch_size'])
        all_classes = np.append(all_classes, add_classes)

    # load net
    num_classes = len(all_classes)

    if net_type == "ConvNet":
        net = ConvNet(len(cifar10_classes))

    model_weights_path = f"weights/{net_type}.pth"

    if os.path.exists(model_weights_path) and use_existing_weights:
        net.load_state_dict(torch.load(model_weights_path))

    if add_superclass:
        model_weights_path = f"weights/{net_type}_100.pth"

        for param in net.parameters():
            param.requires_grad = False

        if net_type == "ConvNet":
            net.fc1.requires_grad_()
            net.fc2.requires_grad_()
            net.fc3 = nn.Linear(net.fc3.in_features, num_classes)

        if not train_model and os.path.exists(model_weights_path) and use_existing_weights:
            net.load_state_dict(torch.load(model_weights_path))

    net.to(device)

    # train
    if train_model:
        losses, accs = train(net, train_loader, val_loader,
                             model_weights_path, device, **params)

    # evaluate
    if evaluate_model:
        train_mat = evaluate(net, train_loader, all_classes, device, "Train set")
        test_mat = evaluate(net, test_loader, all_classes, device, "Test set")

    # visualize
    if train_model:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

        ax1.set_title('Loss')
        ax1.plot(losses, '-x')

        ax2.set_title('Accuracy')
        ax2.plot(accs, '-x')

        if save_plots:
            filename: str = f"out/{net_type}.png"
            if add_superclass:
                filename = f"out/{net_type}_100.png"
            plt.savefig(filename)

        plt.show()

    if evaluate_model:
        filename: str = f"out/{net_type}_cm"
        if add_superclass:
            filename = f"out/{net_type}_100_cm"

        filename_train = filename + "_train.png"
        plt.matshow(train_mat)
        plt.title('Train')
        plt.colorbar()
        if save_plots:
            plt.savefig(filename_train)

        filename_test = filename + "_test.png"
        plt.matshow(test_mat)
        plt.title('Test')
        plt.colorbar()
        if save_plots:
            plt.savefig(filename_test)

        plt.show()


