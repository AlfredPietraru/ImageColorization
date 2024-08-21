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


def yuv_to_rgb(y : torch.Tensor, u : torch.Tensor, v : torch.Tensor):
    y = y - 16
    u = u - 128
    v = v - 128
    B = 1.164 * y + 2.018 * u
    G = 1.164 * y - 0.813 * v - 0.391 * u
    R = 1.164 * y + 1.596 * v
    R = R.byte()
    B = B.byte()
    G = G.byte()
    return torch.stack([R, G, B], dim=-1)

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
    np.savetxt('try.txt', np.round(u.detach().numpy(), 3))
    v = pad_matrix(torch.squeeze(v))
    return yuv_to_rgb(img, u, v)

if __name__ == "__main__":
    model = DeepColorization()
    model.load_state_dict(torch.load("./model.pth"))    
    final_image = evaluation(model)
    cv2.imshow("gata", final_image.numpy())
    cv2.waitKey(0)














