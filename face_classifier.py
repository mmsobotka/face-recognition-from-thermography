import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random


def view_image(train_image):
    plt.imshow(train_image, cmap='gray')
    plt.show()


def prepare_data():
    dirs = os.listdir("data")
    persons_images = [[]]
    labels_images_list = []
    index = 0
    n_train_images = 0
    n_validate_images = 0
    ratio_data_training_vs_validation = 80

    number_of_images = []

    for directory in dirs:

        n_images_person = 0
        for file in os.listdir('data/' + directory):
            image = plt.imread('data/' + directory + '/' + file)
            image = np.reshape(image, (240, 320, 1))
            persons_images[index].append(image)

            n_images_person += 1
            labels_images_list.append(index)

        n_train_images_person = int(n_images_person * ratio_data_training_vs_validation / 100)
        n_validation_images_person = n_images_person - n_train_images_person
        number_of_images.append((n_train_images_person, n_validation_images_person))

        n_train_images += n_train_images_person
        n_validate_images += n_validation_images_person

        index += 1
        persons_images.append([])

    persons_images.pop()

    train_images = np.empty((n_train_images, 240, 320, 1), dtype='uint8')
    validate_images = np.empty((n_validate_images, 240, 320, 1), dtype='uint8')
    train_labels = np.empty((n_train_images, 1), dtype='uint8')
    validate_labels = np.empty((n_validate_images, 1), dtype='uint8')

    index_train_images = 0
    index_validate_images = 0
    for index_person, person_images in enumerate(persons_images):

        random.shuffle(person_images)

        n_train_images_person, n_validate_images_person = number_of_images[index_person]
        for index_image, person_image in enumerate(person_images):
            if index_image < n_train_images_person:
                train_images[index_train_images] = person_image
                train_labels[index_train_images] = index_person
                index_train_images += 1
            else:
                validate_images[index_validate_images] = person_image
                validate_labels[index_validate_images] = index_person
                index_validate_images += 1

    train_images, test_images = train_images / 255.0, validate_images / 255.0

    return (train_images, train_labels), (validate_images, validate_labels)


def view_example_data(train_images, train_labels):
    class_names = ['Mateusz W', 'Agnieszka C', 'Milena S', 'Aleksandra T', 'Klaudia S',
                   'Agata P', 'Natalia R', 'Adriana Z', 'Florian G', 'Marcin K', 'Karol L', 'Michal S', 'Michal K',
                   'Nikola L']

    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap='gray')

        plt.xlabel(class_names[train_labels[i][0]])
    plt.show()
