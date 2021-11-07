import numpy as np
import os
import math


header_1st_frame_len = 1372
header_frame_len = 1372
footer_len = 3860 + 32
image_data_len = (320 * 240 * 2)
first_frame_len = 1372 + (320 * 240 * 2) + 3860
frame_len = 1372 + (320 * 240 * 2) + 3860 + 32
image_size = 320 * 240

C1 = 21764040
C2 = 3033.3
C3 = 134.06


def convert_to_numpy(file, every=1, temp=False):
    """ Read data from a file and convert to numpy array
    """

    no_of_frames = int(os.stat(file).st_size / frame_len)
    print("Number of frames: " + str(no_of_frames))

    index = 0
    seq = []
    with open(file, 'rb') as fi:
        for i in range(0, no_of_frames, every):
            if index == 0:
                fi.read(header_1st_frame_len)
                image_data = fi.read(image_data_len)
                fi.read(footer_len)

            else:
                if every > 1:
                    for i in range(1, every):
                        fi.read(frame_len)
                index = index + (every - 1)
                fi.read(header_frame_len)
                image_data = fi.read(image_data_len)
                fi.read(footer_len)

            res = np.empty([image_size], dtype=int)

            for p in range(0, image_data_len, 2):
                result = int.from_bytes(image_data[p:p + 2], byteorder='little', signed=False)
                if temp:
                    result_temp = (C2 / math.log(C3 + C1 / result) - 273.15) * 10
                else:
                    result_temp = result
                res[int(p / 2)] = result_temp

            seq.append(res)
            index = index + 1

    return seq


def change_range(input_array: np.array):
    image = input_array - input_array.min()
    image = image * 255. / image.max()
    return image
