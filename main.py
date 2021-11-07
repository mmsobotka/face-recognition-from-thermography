from A320G import *
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
    read_videos_save_frames_to_jpg()
