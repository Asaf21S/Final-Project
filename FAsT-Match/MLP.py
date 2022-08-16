import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd


def import_data(data_folder):
    # import the data from the csv files
    all_names = np.array([])  # samples' names
    all_ideal_th = np.array([])  # samples' ideal threshold
    all_features = np.zeros((0, 37))  # samples' features
    for file_name in listdir(data_folder):
        if file_name.endswith(".csv") and file_name.startswith("Training_Data"):
            file_names = np.loadtxt(data_folder + "\\" + file_name, dtype='str', delimiter=',', usecols=0)
            file_ideal_th = np.loadtxt(data_folder + "\\" + file_name, dtype='float', delimiter=',', usecols=1)
            file_features = np.loadtxt(data_folder + "\\" + file_name, dtype='float', delimiter=',',
                                       usecols=tuple(range(2, 39)))
            all_names = np.concatenate((all_names, file_names))
            all_ideal_th = np.concatenate((all_ideal_th, file_ideal_th))
            all_features = np.concatenate((all_features, file_features))
    return all_names, all_ideal_th, all_features


def import_histogram(data_folder):
    # import the histograms from the csv files
    all_names = np.array([])  # samples' names
    all_edges = np.zeros((0, 2))  # samples' min and max distances
    all_bins = np.zeros((0, 100), dtype=int)  # samples' bin values
    for file_name in listdir(data_folder):
        if file_name.endswith(".csv") and file_name.startswith("Histogram_Data"):
            file_names = np.loadtxt(data_folder + "\\" + file_name, dtype='str', delimiter=',', usecols=0)
            file_edges = np.loadtxt(data_folder + "\\" + file_name, dtype='float', delimiter=',', usecols=(1, 2))
            file_bins = np.loadtxt(data_folder + "\\" + file_name, dtype='int', delimiter=',',
                                   usecols=tuple(range(3, 103)))
            all_names = np.concatenate((all_names, file_names))
            all_edges = np.concatenate((all_edges, file_edges))
            all_bins = np.concatenate((all_bins, file_bins))
    return all_names, all_edges, all_bins


def preprocess(features, ideal_th, factor, bins):
    processed_features = np.copy(features)
    processed_ideal_th = np.copy(ideal_th)
    processed_bins = np.copy(bins)

    # Choose relevant features
    processed_features = processed_features[:, (2, 3, 4, 5, 6, 7, 8, 9, 29, 30)]
    # Normalize some of them
    processed_features[:, (0, 9)] = normalize(processed_features[:, (0, 9)])

    processed_ideal_th -= processed_features[:, 5]
    processed_ideal_th *= factor / 2.0
    processed_ideal_th += processed_features[:, 5]
    processed_ideal_th = np.expand_dims(ideal_th, axis=1)

    for ind in range(1, 100):
        processed_bins[:, ind] += processed_bins[:, ind - 1]

    return processed_features, processed_ideal_th, processed_bins


class dataset(Dataset):
    def __init__(self, X, y):
        if not torch.is_tensor(X) and not torch.is_tensor(y):
            self.X = torch.from_numpy(X)
            self.X = self.X.float()
            self.y = torch.from_numpy(y)
            self.y = self.y.float()

    def __getitem__(self, i):
        return self.X[i], self.y[i]

    def __len__(self):
        return len(self.X)


class MLP(nn.Module):
    """
        Multilayer Perceptron for regression.
    """

    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(10, 500)
        self.lin2 = nn.Linear(500, 1000)
        self.lin3 = nn.Linear(1000, 100)
        self.lin4 = nn.Linear(100, 1)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.lin1(x))
        x = self.activation(self.lin2(x))
        x = self.activation(self.lin3(x))
        x = self.lin4(x)
        return x


def train(model, dataloader, loss_function, optimizer):
    loss_epoch = []
    model.train()

    for index_batch, (x, y) in enumerate(dataloader):
        # prediction
        prediction = model(x)

        # loss
        loss = loss_function(prediction, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_epoch.append(loss.item())

    loss_mean_epoch = np.mean(loss_epoch)
    loss_std_epoch = np.std(loss_epoch)
    loss = {'mean': loss_mean_epoch, 'std': loss_std_epoch}
    return loss


def test(model, dataloader, loss_function):
    loss_epoch = []
    model.eval()

    for index_batch, (x, y) in enumerate(dataloader):
        # prediction
        prediction = model(x)

        # loss
        loss = loss_function(prediction, y)
        loss_epoch.append(loss.item())

    loss_mean_epoch = np.mean(loss_epoch)
    loss_std_epoch = np.std(loss_epoch)
    loss = {'mean': loss_mean_epoch, 'std': loss_std_epoch}
    return loss


def train_model(X, y):
    number_epoch = 100
    size_minibatch = 100
    learning_rate = 0.00001
    weight_decay = 0.000001

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    dataset_train = dataset(X_train, y_train)
    dataset_test = dataset(X_test, y_test)
    dataloader_train = DataLoader(dataset_train, batch_size=size_minibatch, shuffle=True, drop_last=False)
    dataloader_test = DataLoader(dataset_test, batch_size=size_minibatch, shuffle=True, drop_last=False)

    mlp = MLP()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate, weight_decay=weight_decay)

    loss_mean_train = np.zeros(number_epoch)
    loss_std_train = np.zeros(number_epoch)
    loss_mean_test = np.zeros(number_epoch)
    loss_std_test = np.zeros(number_epoch)

    min_loss = 1.0

    for i in range(number_epoch):
        loss_train = train(mlp, dataloader_train, loss_function, optimizer)

        loss_mean_train[i] = loss_train['mean']
        loss_std_train[i] = loss_train['std']

        loss_test = test(mlp, dataloader_test, loss_function)

        loss_mean_test[i] = loss_test['mean']
        loss_std_test[i] = loss_test['std']

        if loss_mean_test[i] < min_loss:
            min_loss = loss_mean_test[i]
            torch.save(mlp.state_dict(), './mlp.pth')

    mlp = MLP()
    mlp.load_state_dict(torch.load('./mlp.pth', map_location=torch.device('cpu')))

    return mlp, loss_mean_train, loss_std_train, loss_mean_test, loss_std_test


def load_model(path):
    mlp = MLP()
    mlp.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    return mlp


def model_prediction(model, features):
    x = np.expand_dims(features, axis=0)
    x = torch.from_numpy(x).float()
    y = model(x)
    y = y.cpu().data.numpy()
    y = np.squeeze(y)

    print("MLP prediction for input {}: {}".format(x, y))
    return y


def safety_window(delta):
    p0 = 0.1341
    p1 = 0.0278
    safety = 0.02
    return p0 * delta + p1 - safety


def display_score(model, X, y, factor, set_name):
    y = np.squeeze(y)
    deltas = X[:, 5] + safety_window(X[:, 4])
    X_torch = torch.from_numpy(X).float()
    prediction = model(X_torch)
    prediction = prediction.cpu().data.numpy()
    prediction = np.squeeze(prediction)
    ground_truth = X[:, 5] + (y - X[:, 5]) / (factor * 2)

    indices = np.argsort(y)
    deltas = deltas[indices]
    prediction = prediction[indices]
    ground_truth = ground_truth[indices]
    y = y[indices]

    plt.figure(figsize=(24, 11))
    plt.scatter(range(len(prediction)), ground_truth, s=1, color='lime', label='ground truth')
    plt.scatter(range(len(prediction)), y, s=1, color='green', label='ideal_th')
    plt.scatter(range(len(prediction)), deltas, s=1, color='cyan', label='current')
    plt.scatter(range(len(prediction)), prediction, s=1, color='red', label='prediction')
    plt.scatter([-1000], [y.mean()], s=100, color='blue', label='average ideal_th')
    # ax = plt.gca()
    # ax.set_ylim([0, 0.3])
    plt.legend()

    score = 1 - np.sum(np.square(y - prediction)) / np.sum(np.square(y - y.mean()))
    set_name = set_name + " score: " + str(score)
    plt.title(set_name)

    print("  Score:\t", score)
    accuracy = np.count_nonzero(prediction > ground_truth) / len(prediction)
    current_accuracy = np.count_nonzero(deltas > ground_truth) / len(prediction)
    print("  Accuracy:\t", accuracy)
    print("  Current accuracy:\t", current_accuracy)

    plt.tight_layout()
    plt.show()


def accuracy1(prediction, ground_truth):
    """
    whether ground truth has passed
    """
    return prediction > ground_truth


def accuracy2_by_name(name, prediction, names_hist, edges, bins):
    """
    percentage of passed configurations
    """
    ind = np.where(names_hist == name)[0][0]
    percentage_value = 100 * (prediction - edges[ind, 0]) / (edges[ind, 1] - edges[ind, 0])
    bin_number = int(percentage_value)
    amount = 0
    if bin_number != 0:
        amount = bins[ind, bin_number - 1]
    amount += (percentage_value - bin_number) * (bins[ind, bin_number] - bins[ind, bin_number - 1])
    return 100 * amount / bins[ind, -1]


def accuracy2_by_index(index, prediction, edges, bins):
    """
    percentage of passed configurations
    """
    percentage_value = 100 * (prediction - edges[index, 0]) / (edges[index, 1] - edges[index, 0])
    bin_number = int(percentage_value)
    amount = 0
    if bin_number != 0:
        amount = bins[index, bin_number - 1]
    amount += (percentage_value - bin_number) * (bins[index, bin_number] - bins[index, bin_number - 1])
    return 100 * amount / bins[index, -1]


# plot train and test metric along epochs
def plot_curve_error(train_mean, train_std, test_mean, test_std, x_label, y_label, title):
    plt.figure(figsize=(10, 8))
    plt.title(title)

    alpha = 0.1

    plt.plot(range(len(train_mean)), train_mean, '-', color='red', label='train')
    plt.fill_between(range(len(train_mean)), train_mean - train_std, train_mean + train_std, facecolor='red',
                     alpha=alpha)

    plt.plot(range(len(test_mean)), test_mean, '-', color='blue', label='test')
    plt.fill_between(range(len(test_mean)), test_mean - test_std, test_mean + test_std, facecolor='blue', alpha=alpha)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_curves(model, factor, X_train, X_test, y_train, y_test,
                loss_mean_train, loss_std_train, loss_mean_test, loss_std_test):
    plot_curve_error(loss_mean_train, loss_std_train, loss_mean_test, loss_std_test, 'epoch', 'losses', 'LOSS')

    print("Train:")
    print("  Loss:   \t", loss_mean_train[-1])
    display_score(model, np.copy(X_train), np.copy(y_train), factor, "Train")

    print("Test:")
    print("  Loss:   \t", loss_mean_test[-1])
    display_score(model, np.copy(X_test), np.copy(y_test), factor, "Test")


def plot_heatmaps(model, X, y):
    df = pd.DataFrame(np.append(y, X, axis=1))
    corr = df.corr()
    corr.style.background_gradient(cmap='coolwarm')
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(corr)
    plt.clim(-1, 1)
    plt.colorbar()
    plt.title("ideal_th + features correlation")

    prediction_all = model(torch.from_numpy(X).float())
    prediction_all = prediction_all.cpu().data.numpy()
    df2 = pd.DataFrame(np.append(prediction_all, X, axis=1))
    corr2 = df2.corr()
    corr2.style.background_gradient(cmap='coolwarm')
    plt.subplot(1, 2, 2)
    plt.imshow(corr2)
    plt.clim(-1, 1)
    plt.colorbar()
    plt.title("prediction + features correlation")

    print(corr)
    print(corr2)

    plt.show()
