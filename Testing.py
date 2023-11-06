# simpleCamTest.py
import multiprocessing
import os
import time

import numpy
import numpy as np
import cv2

# from app.image_converter import face_detect, setting, saliency, image_magick

faceCascade = cv2.CascadeClassifier('./haar/haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('./haar/haarcascade_eye.xml')
REDUCE_MAX_RATE = 1.5
MIN_THRESHOLD = 0.2


def resize(img, target_size):
    width = len(img[0])
    height = len(img)
    resize_rate = 0
    # 원본 이미지의 가로 세로 중 작은 값을 최소 값 (img_min_length)으로 잡는다.
    img_max_length = width
    if img_max_length < height:
        img_max_length = height

    # 타겟 사이즈의 가로 세로 중 작은 값을 최소 값 (target_min_length)으로 잡는다.
    target_max_length = target_size[0]
    if target_max_length > target_size[1]:
        target_max_length = target_size[1]

    # 타겟 최소 * REDUCE_MAX_RATE = min_length

    max_length = target_max_length * REDUCE_MAX_RATE
    base_pic = np.zeros((target_size[1], target_size[0], 3), np.uint8)

    h, w = img.shape[:2]
    ash = target_size[1] / h
    asw = target_size[0] / w
    if asw < ash:
        resize_rate = asw
    else:
        resize_rate = ash
    sizeas = (int(w * resize_rate), int(h * resize_rate))
    pic1 = cv2.resize(img, dsize=sizeas)
    base_pic[int(target_size[1] / 2 - sizeas[1] / 2):int(target_size[1] / 2 + sizeas[1] / 2),
    int(target_size[0] / 2 - sizeas[0] / 2):int(target_size[0] / 2 + sizeas[0] / 2), :] = pic1
    # 타겟에 대해 100% 맞추는건 아닌거같다
    # 즉 최소 타겟 사이즈의 1.5배 보다 실제 이미지의 최소 사이즈가 클경우 리사이즈 수행
    if img_max_length > max_length:
        width_rate = target_size[0] / width
        height_rate = target_size[1] / height
        resize_rate = img_max_length / max_length
        new_img = cv2.resize(img, dsize=(int(width * width_rate), int(height * height_rate)))
        return new_img, resize_rate

    else:
        return img, 1


# def revision_crop(crop_point, resize_rate, org_width, org_height):
#     target_size = setting.get_target_size("NEWS", "MOBILE")
#     width_threshold, height_threshold = setting.get_threshold_size("NEWS", "MOBILE")
#     output_path = './'
#
#     thumbnail_size = setting.get_target_split_size("NEWS", "MOBILE")
#
#     # 크롭 포인트에 대한 정보
#     s_width = crop_point[2] - crop_point[0]
#     s_height = crop_point[3] - crop_point[1]
#     t_rate = (1.0 * thumbnail_size[0]) / thumbnail_size[1]
#     s_rate = 1.0 * s_width / s_height
#
#     if t_rate < s_rate:
#         fix_height = s_width * thumbnail_size[1] / thumbnail_size[0] - s_height
#         delta = fix_height / 2
#         y_min = crop_point[1] - delta
#         y_max = crop_point[3] + delta
#         if y_min < 0:
#             y_max -= y_min
#             y_min = 0
#         elif y_max > s_width * t_rate:
#             y_max = s_width * t_rate + y_min
#         crop_point = [crop_point[0], y_min, crop_point[2], y_max]
#     else:
#         fix_width = s_height * thumbnail_size[0] / thumbnail_size[1] - s_width
#
#         delta = fix_width / 2
#         x_min = crop_point[0] - delta
#         x_max = crop_point[2] + delta
#         if x_min < 0:
#             x_max -= x_min
#             x_min = 0
#         elif x_max > s_height * t_rate:
#             x_max = s_height * t_rate + x_min
#         crop_point = [x_min, crop_point[1], x_max, crop_point[3]]
#
#     if resize_rate != 1:
#         crop_point = [int(crop_point[0] / resize_rate), int(crop_point[1] / resize_rate),
#                       int(crop_point[2] / resize_rate), int(crop_point[3] / resize_rate)]
#
#     if crop_point[0] < 0:
#         crop_point[2] -= crop_point[0]
#     if crop_point[2] > org_width:
#         crop_point[0] -= crop_point[2] - org_width
#
#     if crop_point[1] < 0:
#         crop_point[3] -= crop_point[1]
#     if crop_point[3] > org_height:
#         crop_point[1] -= crop_point[3] - org_height
#
#     if crop_point[0] < 0:
#         crop_point[0] = 0
#     if crop_point[1] < 0:
#         crop_point[1] = 0
#     if crop_point[2] > org_width:
#         crop_point[2] = org_width
#     if crop_point[3] > org_height:
#         crop_point[3] = org_height
#
#     new_width = crop_point[2] - crop_point[0]
#     new_height = crop_point[3] - crop_point[1]
#
#     # 가로가 긴 경우
#     # 100 , 50 ->
#     if new_height < new_width:
#         ratio = 1.0 * thumbnail_size[0] / org_width
#     else:
#         ratio = 1.0 * thumbnail_size[1] / org_height
#
#     crop_width = 1.0 * new_width * ratio
#
#     # 세로가 긴 직사가형이고, crop 후에 가로가 40 미만일 경우
#     if org_height > org_width and crop_width < width_threshold:
#         image_magick.make_crop(file_path, './test/23.jpeg', (0, 0), org_width, org_width, target_size)
#     else:
#         image_magick.make_crop(file_path, './test/23.jpeg', (crop_point[0], crop_point[1]), new_width, new_height,
#                                target_size)
#
#     return "OK"


def get_threshold(img):
    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    if img.ndim < 3:
        return MIN_THRESHOLD
    hist_list = []
    try:
        for ch, col in enumerate(color):
            hist_item = cv2.calcHist([img], [ch], None, [256], [0, 256])
            cv2.normalize(hist_item, hist_item, 0, 255, cv2.NORM_MINMAX)
            hist = numpy.int32(numpy.around(hist_item))
            hist_list.append(hist)

        standard_deviation_sum = 0
        for hist in hist_list:
            standard_deviation_sum += numpy.sqrt(numpy.sum((hist - numpy.average(hist)) ** 2) / len(hist))
    except Exception as msg:
        # print msg
        return MIN_THRESHOLD

        # 0.58, 170의 뜻??
    DEVIATION = standard_deviation_sum * 0.58 / 170
    # DEVIATION = standard_deviation_sum * 0.55 / 170
    if DEVIATION > 0.52:
        DEVIATION = 0.52
    return DEVIATION


def test(file):
    output_path = "./crop"

    # 파일 목록 출력
    start = time.time()
    print(file)
    img = cv2.imread(file)
    if img is None:
        pass
    target_size = [78, 78]
    image, resize_rate = resize(img, target_size)
    rgb_result = 0
    b, g, r = cv2.split(img)
    height_middle = len(img) / 2
    width_middle = len(img[0]) / 2

    target_list = []
    shift_count = 33
    for i in range(4):
        row_start_index = int(i / 2 * height_middle)
        row_end_index = int(row_start_index + height_middle)
        col_start_index = int(i % 2 * width_middle)
        col_end_index = int(col_start_index + width_middle)
        for j in range(3):
            if j % 3 == 0:
                target_list = r
            elif j % 3 == 1:
                target_list = g
            elif j % 3 == 2:
                target_list = b
            STEP = 2
            target_list = target_list[row_start_index:row_end_index:STEP, col_start_index: col_end_index: STEP]
            # AND_MASK = 32
            target_list = target_list / 32
            list_sum = int(target_list.sum())
            count = 1
            for i in target_list.shape:
                count = count * i
            if count == 1:
                rgb_result += 0
            else:
                rgb_result += (int(list_sum / count) << shift_count)
            shift_count -= 3
    rgb_result = "%032x" % rgb_result
    cv2.imwrite(os.path.join(output_path + file[6:]), image)

    end = time.time()
    print("image out : " + output_path + file[6:] + "\t hashKey : " + rgb_result + "\t elapsedTime : " + str(
        end - start))


if __name__ == '__main__':
    start = time.time()
    pool = multiprocessing.Pool(processes=4)
    directory_path = "./test1"
    all_files = [os.path.join(directory_path, file) for file in os.listdir(directory_path)]
    #
    pool.map(test, all_files)
    pool.close()
    pool.join()
    for i in all_files:
        test(i)
    end = time.time()

    print("수행시간: %f 초" % (end - start))
