import cv2
import os

SIZE = (512, 512)
PATH_IMAGES = "training_images/initial_images/"
PATH_COLOR_IMAGES = "training_images/preprocessed_images/"
PATH_GRAY_IMAGES = "training_images/gray_scale_images/"
PATH_GRAY_EVALUATION_IMAGES = "evaluation_images/gray_scale_images/"
PATH_COLOR_EVALUATION_IMAGES = "evaluation_images/color_images/"

def new_color_img_path(i : int):
    return PATH_COLOR_IMAGES + "color_image" + str(i) + ".jpg"

def new_gray_img_path(i : int):
    return PATH_GRAY_IMAGES + "gray_image" + str(i) + ".jpg"

def return_gray_image(i : int):
    return cv2.imread(new_gray_img_path(i), cv2.IMREAD_GRAYSCALE)

def return_color_image(i : int):
    return cv2.imread(new_color_img_path(i), cv2.IMREAD_COLOR)

def return_yuv_image(i : int):
    img_yuv = cv2.cvtColor(return_color_image(i), cv2.COLOR_BGR2YUV)
    return cv2.split(img_yuv)




# for i in range(88, 496, 1):
#     image_name =  "image" + str(i) + ".png"
#     image_path = PATH_IMAGES + image_name
#     img = cv2.imread(image_path, cv2.IMREAD_COLOR)
#     resized_img = cv2.resize(img, SIZE)
#     gray_image = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)
#     cv2.imwrite(new_color_img_path(i), resized_img)
#     cv2.imwrite(new_gray_img_path(i), gray_image)

def get_current_img_path(idx : int):
    if (idx <= 99):
        value = "00000" + str(idx) + ".png"
    if (idx >= 100):
        value = "0000" + str(idx) + ".png"
    return PATH_COLOR_IMAGES + value

def get_new_name(idx : int):
    return PATH_COLOR_IMAGES + "color_image" + str(idx) + ".jpg"

# files = os.listdir(PATH_COLOR_IMAGES)
# for i in range(62, 495, 1):
#     os.rename(get_new_name(i), get_new_name(i-1))

# for i in range(88, 496, 1):
#      os.rename(get_current_img_path(i), get_new_name(i))


for i in range(1, 468, 1):
    image_path = new_color_img_path(i)
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imwrite(new_gray_img_path(i), gray_image)
