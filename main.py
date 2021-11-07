from A320G import *
from face_classifier import *
import cv2


def read_videos_save_frames_to_jpg():
    files = os.listdir("A320G")
    for file in files:
        print(file)

    thermal_images = 1

    for file in files:
        filename = "A320G/" + file
        dataset = convert_to_numpy(filename, every=1)
        for x in range(len(dataset)):
            dataset[x] = dataset[x].reshape((240, 320))

        dataset_array = np.stack(dataset)
        directory_name = f'data/person{thermal_images}'
        try:
            os.mkdir(directory_name)
            for idx in range(len(dataset_array)):
                cv2.imwrite(f'{directory_name}/im{idx}.jpg', change_range(dataset_array[idx]))
        except Exception:
            print('something went wrong')
        thermal_images += 1


if __name__ == "__main__":
    # read_videos_save_frames_to_jpg()

    (train_images, train_labels), (test_images, test_labels) = prepare_data()
    view_example_data(train_images, train_labels)

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(32, (7, 7), activation='relu', input_shape=(240, 320, 1)))
    model.add(tf.keras.layers.MaxPooling2D((4, 4)))
    model.add(tf.keras.layers.Conv2D(64, (7, 7), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((4, 4)))
    model.add(tf.keras.layers.Conv2D(64, (7, 7), activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(14, activation='softmax'))

    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_images, train_labels, batch_size=32, epochs=6,
                        validation_data=(test_images, test_labels))

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(test_acc)
