import cv2
import torch
import training as tr 
import preprocessing as prep
from deep_colorization import DeepColorization 
from skimage.feature import daisy

def pad_matrix(matrix):
    return torch.nn.functional.pad(matrix, 
                            (3, 3, 3, 3), mode='constant', value=0)


def yuv_to_rgb(y, u, v):
    u = u * 255 - 128.0
    v = v * 255 - 128.0

    r = y + 1.402 * v
    g = y - 0.344136 * u - 0.714136 * v
    b = y + 1.772 * u


    rgb = torch.stack([r, g, b], dim=-1)
    rgb = torch.clamp(rgb, 0, 255).byte()
    return rgb

def evaluation(model):
    model.eval()
    img = torch.Tensor(prep.return_gray_image(8))
    features = torch.zeros(pow(tr.TOTAL_SIZE, 2), 81)
    descriptors = torch.Tensor(daisy(img, step=1, radius=1, histograms=3, orientations=8, rings=1))
    index = 0
    for i in range(tr.MINIM_INDEX, tr.MAXIM_INDEX + 1, 1):
        for j in range(tr.MINIM_INDEX, tr.MAXIM_INDEX + 1, 1):
            window = [i - tr.MINIM_INDEX, j - tr.MINIM_INDEX, 
                      i + tr.MINIM_INDEX - 1, j + tr.MINIM_INDEX - 1]
            sub_area = img[window[0]:window[2], window[1]:window[3]
                            ].reshape(tr.LOW_FEATURE_SIZE)
            features[index] = torch.cat((sub_area, descriptors[i][j]), 0)
            index += 1
    result = torch.Tensor(model(features)).reshape(
        shape=(tr.TOTAL_SIZE, tr.TOTAL_SIZE, 2))
    u,v = result.chunk(2, dim=2)
    u = pad_matrix(torch.squeeze(u))
    v = pad_matrix(torch.squeeze(v))
    return yuv_to_rgb(img, u, v)

if __name__ == "__main__":
    model = DeepColorization()
    model.load_state_dict(torch.load("./model.pth"))    
    final_image = evaluation(model)
    cv2.imshow("gata", final_image.numpy())
    cv2.waitKey(0)














