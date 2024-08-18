import cv2

SIZE = (512, 512)
PATH_IMAGES = "training_images/initial_images/"
PATH_COLOR_IMAGES = "training_images/preprocessed_images/"
PATH_GRAY_IMAGES = "training_images/gray_scale_images/"

def new_color_img_path(i : int):
    return PATH_COLOR_IMAGES + "color_image" + str(i) + ".jpg"

def new_gray_img_path(i : int):
    return PATH_GRAY_IMAGES + "gray_image" + str(i) + ".jpg"

for i in range(1, 9, 1):
    image_name =  "image" + str(i) + ".jpg"
    image_path = PATH_IMAGES + image_name
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    resized_img = cv2.resize(img, SIZE)
    gray_image = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(new_color_img_path(i), resized_img)
    cv2.imwrite(new_gray_img_path(i), gray_image)

