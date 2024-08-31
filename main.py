import cv2
import torch
import preprocessing as prep
from deep_colorization import DeepColorization
from skimage.feature import daisy

SIZE = 512
LOW_FEATURE_SIZE = 49
MINIM_INDEX = 4
MAXIM_INDEX = SIZE - 3
TOTAL_SIZE = SIZE - 6
FEATURES = torch.zeros(TOTAL_SIZE, 81)
OUTPUT = torch.zeros(TOTAL_SIZE, TOTAL_SIZE, 2)
model = DeepColorization()

def pad_matrix(matrix):
    return torch.nn.functional.pad(matrix, 
                            (3, 3, 3, 3), mode='constant', value=0)


def extract_model_result():
    u,v = OUTPUT.chunk(2, dim=2)
    u = pad_matrix(torch.squeeze(u))
    v = pad_matrix(torch.squeeze(v))
    return u, v


def yuv_to_rgb(y : torch.Tensor, u : torch.Tensor, v : torch.Tensor):
    u = u * 256
    v = v * 256
    R = y + 1.4075 * (v - 128) 
    G = y - 0.3455 * (u - 128) - 0.7169 * (v - 128)
    B = y + 1.7790 * (u - 128)
    return torch.stack([R.to(torch.uint8), G.to(torch.uint8), B.to(torch.uint8)], dim=-1)

def evaluation_one_image(idx : int):
    img = cv2.resize(prep.return_gray_image(idx),(SIZE, SIZE))
    img = torch.Tensor(img)
    descriptors = torch.Tensor(daisy(img, step=1, radius=1, histograms=3, orientations=8, rings=1))
    for i in range(MINIM_INDEX, MAXIM_INDEX + 1, 1):
        for j in range(MINIM_INDEX, MAXIM_INDEX + 1, 1):
            window = [i - MINIM_INDEX, j - MINIM_INDEX, 
                      i + MINIM_INDEX - 1, j + MINIM_INDEX - 1]
            sub_area = img[window[0]:window[2], window[1]:window[3]
                            ].reshape(LOW_FEATURE_SIZE)
            FEATURES[j - MINIM_INDEX] = torch.cat((sub_area, descriptors[i][j]), 0)
        OUTPUT[i - MINIM_INDEX] = model(FEATURES)
    u, v = extract_model_result()
    rgb_image = yuv_to_rgb(img, u, v)
    return cv2.ximgproc.jointBilateralFilter(
        cv2.resize(prep.return_gray_image(idx),(SIZE, SIZE)),
         rgb_image.numpy(), 10, sigmaColor=15, sigmaSpace=15)

def evaluate(start_idx : int, end_idx : int, display : bool):
    for i in range(start_idx, end_idx, 1):
        final_img = evaluation_one_image(i)
        if display:
            cv2.imshow("final", final_img)
            cv2.waitKey(0)
        else :    
            cv2.imwrite(prep.return_evaluation_path(i), final_img) 

if __name__ == "__main__":
    model.load_state_dict(torch.load("./model.pth"))
    model.eval()
    evaluate(466, 467, True)

