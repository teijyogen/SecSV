import numpy as np
from torchvision import datasets, transforms

def extr_noniid_dirt(train_dataset, test_dataset, num_users, num_classes, alpha=0.5):
    num_imgs_perc_train, num_imgs_train_total = int(len(train_dataset)/num_classes), len(train_dataset)
    num_imgs_perc_test, num_imgs_test_total = int(len(test_dataset)/num_classes), len(test_dataset)
    dict_users_train = {i: np.array([]) for i in range(num_users)}
    dict_users_test = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_imgs_train_total)
    labels = np.array(train_dataset.targets)
    idxs_test = np.arange(num_imgs_test_total)
    labels_test = np.array(test_dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    labels = idxs_labels[1, :]

    idxs_labels_test = np.vstack((idxs_test, labels_test))
    idxs_labels_test = idxs_labels_test[:, idxs_labels_test[1, :].argsort()]
    idxs_test = idxs_labels_test[0, :]
    #print(idxs_labels_test[1, :])

    distribution = np.random.dirichlet(np.repeat(alpha, num_users), size=num_classes)


    # divide and assign
    idxs_classes = []
    for j in range(num_classes):
        idxs_classj = list(idxs[j*num_imgs_perc_train:(j+1)*num_imgs_perc_train])
        idxs_classes.append(idxs_classj)

    idxs_classes_test = []
    for j in range(num_classes):
        idxs_classj_test = list(idxs_test[j*num_imgs_perc_test:(j+1)*num_imgs_perc_test])
        idxs_classes_test.append(idxs_classj_test)

    for i in range(num_users):

        for j in range(num_classes):
            rand_set = np.random.choice(idxs_classes[j], int(distribution[j][i]*num_imgs_perc_train), replace=False)
            idxs_classes[j] = list(set(idxs_classes[j]) - set(rand_set))

            dict_users_train[i] = np.concatenate((dict_users_train[i], rand_set), axis=0)

            rand_set_test = np.random.choice(idxs_classes_test[j], int(distribution[j][i]*num_imgs_perc_test), replace=False)
            idxs_classes_test[j] = list(set(idxs_classes_test[j]) - set(rand_set_test))
            dict_users_test[i] = np.concatenate((dict_users_test[i], rand_set_test), axis=0)

    return dict_users_train, dict_users_test

def get_mnist_iid(num_users):
    apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
    train_data = datasets.MNIST('data', train=True, download=True, transform=apply_transform)
    test_data = datasets.MNIST('data', train=False, download=True, transform=apply_transform)

    indices_train = np.array([i for i in range(len(train_data))])
    indices_test = np.array([i for i in range(len(test_data))])
    np.random.shuffle(indices_train)
    np.random.shuffle(indices_test)

    indices_train_ls = np.array_split(indices_train, num_users)
    indices_test_ls = np.array_split(indices_test, num_users)

    return train_data, test_data, indices_train_ls, indices_test_ls

def get_cifar_iid(num_users):
    apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_data = datasets.CIFAR10('data', train=True, download=True, transform=apply_transform)
    test_data = datasets.CIFAR10('data', train=False, download=True, transform=apply_transform)

    indices_train = np.array([i for i in range(len(train_data))])
    indices_test = np.array([i for i in range(len(test_data))])
    np.random.shuffle(indices_train)
    np.random.shuffle(indices_test)

    indices_train_ls = np.array_split(indices_train, num_users)
    indices_test_ls = np.array_split(indices_test, num_users)

    return train_data, test_data, indices_train_ls, indices_test_ls

def get_mnist_dirt(num_users, alpha=0.5):
    apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
    train_data = datasets.MNIST('data', train=True, download=True, transform=apply_transform)
    test_data = datasets.MNIST('data', train=False, download=True, transform=apply_transform)

    indices_train_ls, indices_test_ls = extr_noniid_dirt(train_data, test_data, num_users, 10, alpha)
    return train_data, test_data, indices_train_ls, indices_test_ls

def get_cifar_dirt(num_users, alpha=0.5):
    apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_data = datasets.CIFAR10('data', train=True, download=True, transform=apply_transform)

    test_data = datasets.CIFAR10('data', train=False, download=True, transform=apply_transform)

    indices_train_ls, indices_test_ls = extr_noniid_dirt(train_data, test_data, num_users, 10, alpha)
    return train_data, test_data, indices_train_ls, indices_test_ls

