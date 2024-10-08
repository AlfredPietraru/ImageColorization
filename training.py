
import torch
from deep_colorization import DeepColorization
import preprocessing as prep
from skimage.feature import daisy
import torch.nn as nn


WINDOW_SIZE = (7, 7)
MINIM_INDEX = WINDOW_SIZE[0] // 2 + 1
MAXIM_INDEX = prep.SIZE[0] - MINIM_INDEX + 1
TOTAL_SIZE = MAXIM_INDEX - MINIM_INDEX + 1
NR_SAMPLED_PIXELS = 2000
LOW_FEATURE_SIZE = 49
U_MAX = 0.436
V_MAX = 0.615
NUMBER_IMAGES = 460
NUMBER_EPOCHS = 10

low_patch_features = torch.zeros([NR_SAMPLED_PIXELS, LOW_FEATURE_SIZE], dtype=torch.int32)
y_values = torch.zeros(size=(NR_SAMPLED_PIXELS, 2))

def get_pixel_coordinates():
    return torch.randint(MINIM_INDEX, MAXIM_INDEX, 
                                      size=(NR_SAMPLED_PIXELS, 2))

def get_window_coordinates(pixels : torch.Tensor):
    return torch.cat((pixels - MINIM_INDEX, pixels + MINIM_INDEX - 1), 1)

def extract_low_features(pixels : torch.Tensor, img : torch.Tensor):
    window_coordinates = get_window_coordinates(pixels)
    for i in range(NR_SAMPLED_PIXELS):
        low_patch_features[i] = img[window_coordinates[i][0]:window_coordinates[i][2],
                                window_coordinates[i][1]:window_coordinates[i][3]
                                ].reshape(LOW_FEATURE_SIZE)
    return low_patch_features

def extract_middle_features(descriptors : torch.Tensor, 
                            pixels: torch.Tensor):
    middle_features = torch.zeros(size=(NR_SAMPLED_PIXELS, 
                                        descriptors.shape[2])).float()
    for i in  range(NR_SAMPLED_PIXELS):
        middle_features[i] = descriptors[pixels[i][0]][pixels[i][1]]
    return middle_features

def merge_features(low_features : torch.Tensor, 
                   middle_features : torch.Tensor):
    return torch.cat((low_features, middle_features), 1)

def create_y_values(pixels : torch.Tensor, i : int):
    y, u, v = prep.return_yuv_image(i)
    u = u - 128
    v = v - 128
    for i in range(NR_SAMPLED_PIXELS):
        y_values[i][0] = u[pixels[i][0]][pixels[i][1]]
        y_values[i][1] = v[pixels[i][0]][pixels[i][1]]
    return y_values / 255

def train_loop(model, optimizer, loss_function):
    model.train()
    for k in range(0, NUMBER_EPOCHS, 1):
        for i in range(1, NUMBER_IMAGES, 1):
            img = torch.Tensor(prep.return_gray_image(i))
            descriptors = torch.Tensor(daisy(img, step=1, radius=1, 
                                             histograms=3, orientations=8, rings=1))
            pixels = get_pixel_coordinates().int()
            x_features = merge_features(extract_low_features(pixels, img), 
                                extract_middle_features(descriptors, pixels))
            loss = loss_function(model(x_features), create_y_values(pixels, i)),
            optimizer.zero_grad()
            loss[0].backward()
            optimizer.step()
            print("total loss is: %f %d", loss[0], i)
        torch.save(model.state_dict(), "./model.pth")
        print("gata o epoca")

if __name__ == "__main__":
    model = DeepColorization()
    model.load_state_dict(torch.load("./model.pth"))
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loop(model, optimizer, loss_function)
