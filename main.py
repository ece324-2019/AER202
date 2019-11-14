import os
import argparse
import torch
from torch.utils.data import DataLoader
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

import models
from utils import process_images


TEST_IMAGE_DIR = 'datasets/test_cropped'
FULL_IMAGE_DIR = 'datasets/full'


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog='AER202', description='Tool that uses machine learning to identify planes from a photo.')
    parser.add_argument('--batch_size', '-b', type=int, default=50,
                        help='Size of batches to use during training')
    parser.add_argument('--learning_rate', '-lr', type=float,
                        default=0.01, help='Learning rate to use during training')
    parser.add_argument('--epochs', '-e', type=int, default=20,
                        help='Number of epochs to use during training')
    parser.add_argument('--eval_every', '-ee', type=int, default=10,
                        help='Number of epochs to wait before evaluating accuracy of model')
    parser.add_argument('--seed', '-s', type=int, default=1,
                        help='Seed to use for any random functions')
    parser.add_argument('--disable_cuda', '-dcuda',
                        action='store_true', help='Disable CUDA')
    parser.add_argument('--process_images', '-pi', action='store_true',
                        help='Prepare images for training by performing actions such as cropping and downscaling')
    parser.add_argument('--small_dataset', '-sd', action='store_true',
                        help='Train and validate on a smaller test dataset for quick testing')
    args = parser.parse_args()

    return args


def get_classes(is_small_dataset: bool):
    """Get names of classes from the immediate subdirectories of the main image directory"""
    images_dir = TEST_IMAGE_DIR if is_small_dataset else FULL_IMAGE_DIR
    classes = next(os.walk(images_dir))[1]

    return classes


def one_hot(x, dim):
    vec = torch.zeros(dim)
    vec[x] = 1.0
    return vec


def load_model_baseline(lr: float, num_classes: int):
    """Load baseline model"""
    model_baseline = models.Baseline(num_classes)
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model_baseline.parameters(), lr)

    return model_baseline, loss_func, optimizer


def evaluate(model, val_loader, disable_cuda):
    total_corr = 0

    for i, batch in enumerate(val_loader):
        features, label = batch
        if torch.cuda.is_available() and not disable_cuda:
            features = features.cuda()
            label = label.cuda()

        # Run model on data
        prediction = model(features)

        # Check number of correct predictions
        torch_max = torch.max(prediction, 1)
        for j in range(prediction.size()[0]):
            if torch_max[1][j].item() == label[j].item():
                total_corr += 1

    return float(total_corr) / len(val_loader.dataset)


def main():
    args = parse_arguments()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.process_images:
        process_images(args.small_dataset)

    # Load data and normalize images
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    dataset = torchvision.datasets.ImageFolder(FULL_IMAGE_DIR, transform=transform)

    # Split data
    train_data, valid_data = train_test_split(dataset, test_size=0.2, shuffle=True, random_state=args.seed)

    # Encode data
    label_encoder = LabelEncoder()
    classes = get_classes(args.small_dataset)
    int_classes = label_encoder.fit_transform(classes)
    oneh_encoder = OneHotEncoder(categories='auto')
    int_classes = int_classes.reshape(-1, 1)
    oneh_labels = oneh_encoder.fit_transform(int_classes).toarray()

    # Create dataloaders
    dataloader_train = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    dataloader_valid = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)

    target_transform = torchvision.transforms.Compose([torchvision.transforms.transforms.Lambda(lambda x: one_hot(x, len(classes)))])

    model_baseline, loss_func, optimizer = load_model_baseline(args.learning_rate, len(classes))
    if torch.cuda.is_available() and not args.disable_cuda:
        model_baseline.cuda()

    # Arrays to keep track of data while training to plot
    gradient_steps = []
    training_accuracies = []
    validation_accuracies = []
    training_losses = []
    validation_losses = []

    for epoch in range(args.epochs):
        accum_loss = 0
        accum_loss_valid = 0
        num_batches = 0
        num_batches_valid = 0

        for i, batch in enumerate(dataloader_train):
            # Get batch of data
            features, label = batch
            if torch.cuda.is_available() and not args.disable_cuda:
                features = features.cuda()
                label = label.cuda()

            num_batches += 1

            # Set gradients to zero
            optimizer.zero_grad()

            # Run neural network on batch
            predictions = model_baseline(features)
            if torch.cuda.is_available() and not args.disable_cuda:
                predictions = predictions.cuda()

            # Compute loss
            target_tensor = torch.Tensor(oneh_labels[label.cpu()])
            if torch.cuda.is_available() and not args.disable_cuda:
                target_tensor = target_tensor.cuda()
            batch_loss = loss_func(input=predictions.squeeze(), target=target_tensor)

            accum_loss += batch_loss

            # Calculate gradients
            batch_loss.backward()
            optimizer.step()

            # gradient_steps.append(t + 1)

        # Calculate validation losses
        for j, batch_valid in enumerate(dataloader_valid):
            num_batches_valid += 1

            # Get batch of data
            features_valid, label_valid = batch_valid
            if torch.cuda.is_available() and not args.disable_cuda:
                features_valid = features_valid.cuda()
                label_valid = label_valid.cuda()

            # Run neural network on validation batch
            predictions_valid = model_baseline(features_valid)
            if torch.cuda.is_available() and not args.disable_cuda:
                predictions_valid = predictions_valid.cuda()

            # Compute loss
            target_tensor = torch.Tensor(oneh_labels[label_valid.cpu()])
            if torch.cuda.is_available() and not args.disable_cuda:
                target_tensor = target_tensor.cuda()
            batch_loss_valid = loss_func(input=predictions_valid.squeeze(),
                                         target=target_tensor)

            accum_loss_valid += batch_loss_valid

        # Store epoch data in lists
        training_losses.append(accum_loss.detach() / num_batches)
        validation_losses.append(accum_loss_valid.detach() / num_batches_valid)
        training_acc = evaluate(model_baseline, dataloader_train, args.disable_cuda)
        training_accuracies.append(training_acc)
        valid_acc = evaluate(model_baseline, dataloader_valid, args.disable_cuda)
        validation_accuracies.append(valid_acc)
        print("epoch:", epoch, "training_acc:", training_acc, "valid_acc:", valid_acc)

    plt.plot(training_accuracies)
    plt.plot(validation_accuracies)
    plt.title('Accuracies')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()

    plt.plot(training_losses)
    plt.plot(validation_losses)
    plt.title('Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


if __name__ == '__main__':
    main()
