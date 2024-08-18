import cv2
import torch
import preprocessing as prep
from skimage.feature import daisy
WINDOW_SIZE = (7, 7)
LOW_FEATURE_SIZE = 49
MINIM_INDEX = WINDOW_SIZE[0] // 2 + 1
NR_SAMPLED_PIXELS = 30

def get_pixel_coordinates():
    # pixel_coordinates = torch.randint(MINIM_INDEX, prep.SIZE[0] - MINIM_INDEX, 
    #                                   size=(NR_SAMPLED_PIXELS, 2))
    # return pixel_coordinates
    return torch.Tensor(
    [
        [4, 4],
        [116,  86],
        [104, 184],
        [378,  65],
        [279, 311],
        [230, 427],
        [327, 308],
        [240, 226],
        [407, 389],
        [297, 147],
        [ 51,  12],
        [ 61, 149],
        [216, 444],
        [499,  99],
        [432, 299],
        [271, 467],
        [231,  19],
        [ 33, 239],
        [352, 430],
        [273, 472],
        [467,  89],
        [408, 229],
        [389, 114],
        [397, 272],
        [216,  11],
        [402, 425],
        [256, 297],
        [ 84, 317],
        [ 72,  76],
        [480, 485],
        [ 47, 104]])

def get_window_coordinates(pixels : torch.Tensor):
    return torch.cat((pixels - MINIM_INDEX, pixels + MINIM_INDEX - 1), 1)

def extract_low_features(window_coordinates : torch.Tensor, img : torch.Tensor):
    low_patch_feature = torch.zeros([NR_SAMPLED_PIXELS, LOW_FEATURE_SIZE], dtype=torch.int32)
    for i in range(NR_SAMPLED_PIXELS):
        low_patch_feature[i] = img[window_coordinates[i][0]:window_coordinates[i][2],
                                window_coordinates[i][1]:window_coordinates[i][3]
                                ].reshape(LOW_FEATURE_SIZE)
    return low_patch_feature

def extract_middle_features(img : torch.Tensor, pixels: torch.Tensor):
    descriptors = daisy(img, step=1, radius=1, histograms=3, orientations=8, rings=1)
    descriptors = torch.Tensor(descriptors)
    middle_features = torch.zeros(size=(NR_SAMPLED_PIXELS, 
                                        descriptors.shape[2])).float()
    for i in  range(NR_SAMPLED_PIXELS):
        middle_features[i] = descriptors[pixels[i][0]][pixels[i][1]]
    return middle_features

def merge_features(low_features : torch.Tensor, 
                   middle_features : torch.Tensor):
    return torch.cat((low_features, middle_features), 1)


img = cv2.imread(prep.new_gray_img_path(1), cv2.IMREAD_GRAYSCALE)
img = torch.Tensor(img)
pixels = get_pixel_coordinates().int()
window_coordinates = get_window_coordinates(pixels).int()

low_features = extract_low_features(window_coordinates, img)
middle_features = extract_middle_features(img, pixels)
input_features = merge_features(low_features, middle_features)

print(input_features.shape)
print(input_features)





