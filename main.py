import cv2
import torch
import training as tr 
import preprocessing as prep
from deep_colorization import DeepColorization 
from skimage.feature import daisy
import numpy as np

def pad_matrix(matrix):
    return torch.nn.functional.pad(matrix, 
                            (3, 3, 3, 3), mode='constant', value=0)


def extract_model_result(y_features : torch.Tensor):
    y_features = y_features.reshape(
        shape=(tr.TOTAL_SIZE, tr.TOTAL_SIZE, 2))
    u,v = y_features.chunk(2, dim=2)
    u = pad_matrix(torch.squeeze(u))
    v = pad_matrix(torch.squeeze(v))
    return u, v
    


def yuv_to_rgb(y : torch.Tensor, u : torch.Tensor, v : torch.Tensor):
    y = y - 16
    u = u - 128
    v = v - 128
    B = 1.164 * y + 2.018 * u
    G = 1.164 * y - 0.813 * v - 0.391 * u
    R = 1.164 * y + 1.596 * v
    return torch.stack([R.byte(), G.byte(), B.byte()], dim=-1)

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
    u, v = extract_model_result(torch.Tensor(model(features)))
    return yuv_to_rgb(img, u, v)

if __name__ == "__main__":
    model = DeepColorization()
    model.load_state_dict(torch.load("./model.pth"))    
    final_image = evaluation(model)
    final_image = cv2.ximgproc.jointBilateralFilter(prep.return_gray_image(8),
                                                     final_image.numpy(), 10, sigmaColor=15, sigmaSpace=15) 
    cv2.imshow("gata", final_image)
    cv2.waitKey(0)














