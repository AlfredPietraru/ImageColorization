
import torch
from deep_colorization import DeepColorization
import preprocessing as prep
from skimage.feature import daisy
import torch.nn as nn


WINDOW_SIZE = (7, 7)
LOW_FEATURE_SIZE = 49
MINIM_INDEX = WINDOW_SIZE[0] // 2 + 1
MAXIM_INDEX = prep.SIZE[0] - MINIM_INDEX + 1
TOTAL_SIZE = MAXIM_INDEX - MINIM_INDEX + 1
NR_SAMPLED_PIXELS = 100

def get_pixel_coordinates():
    return torch.randint(MINIM_INDEX, MAXIM_INDEX, 
                                      size=(NR_SAMPLED_PIXELS, 2))

def get_window_coordinates(pixels : torch.Tensor):
    return torch.cat((pixels - MINIM_INDEX, pixels + MINIM_INDEX - 1), 1)

def extract_low_features(pixels : torch.Tensor, img : torch.Tensor):
    window_coordinates = get_window_coordinates(pixels)
    low_patch_feature = torch.zeros([NR_SAMPLED_PIXELS, LOW_FEATURE_SIZE], dtype=torch.int32)
    for i in range(NR_SAMPLED_PIXELS):
        low_patch_feature[i] = img[window_coordinates[i][0]:window_coordinates[i][2],
                                window_coordinates[i][1]:window_coordinates[i][3]
                                ].reshape(LOW_FEATURE_SIZE)
    return low_patch_feature

def extract_middle_features(pixels: torch.Tensor, img : torch.Tensor):
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

def create_y_values(pixels : torch.Tensor, i : int):
    _, u, v = prep.return_yuv_image(i)
    y_values = torch.zeros(size=(NR_SAMPLED_PIXELS, 2))
    for i in range(NR_SAMPLED_PIXELS):
        y_values[i][0] = u[pixels[i][0]][pixels[i][1]]
        y_values[i][1] = v[pixels[i][0]][pixels[i][1]]
    return y_values / 255

def train_loop(model, optimizer, loss_function):
    model.train()
    for k in range(1, 3, 1):
        for i in range(1, 7, 1):
            for j in range(40):
                img = torch.Tensor(prep.return_gray_image(i))
                pixels = get_pixel_coordinates().int()
                x_features = merge_features(extract_low_features(pixels, img), 
                                    extract_middle_features(pixels, img))
                y_computed = model(x_features)
                y_features = create_y_values(pixels, i)
                loss = loss_function(
                    torch.reshape(y_computed, shape=(1, len(y_computed) * 2)),
                    torch.reshape(y_features,shape=(1, len(y_features) * 2)) 
                        ),
                print(loss[0])
                optimizer.zero_grad()
                loss[0].backward()
                optimizer.step()
            print("a terminat o imagine")
        print("gata o epoca")

if __name__ == "__main__":
    model = DeepColorization()
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loop(model, optimizer, loss_function)
    torch.save(model.state_dict(), "./model.pth")
