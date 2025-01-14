import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from .image_io import crop_patch
from copy import copy
import cv2
from .features_resnet import DeepFeatureExtractor
from torch.nn.functional import interpolate

# used for linear mapping...
def linear_mapping(img):
    return (img - img.min()) / (img.max() - img.min())

# pre-processing the image...
def pre_process(img):
    # get the size of the img...
    height, width = img.shape
    img = np.log(img + 1)
    img = (img - np.mean(img)) / (np.std(img) + 1e-5)
    # use the hanning window...
    window = window_func_2d(height, width)
    img = img * window

    return img

def window_func_2d(height, width):
    win_col = np.hanning(width)
    win_row = np.hanning(height)
    mask_col, mask_row = np.meshgrid(win_col, win_row)

    win = mask_col * mask_row

    return win

def random_warp(img):
    a = -180 / 16
    b = 180 / 16
    r = a + (b - a) * np.random.uniform()
    # rotate the image...
    matrix_rot = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), r, 1)
    img_rot = cv2.warpAffine(np.uint8(img * 255), matrix_rot, (img.shape[1], img.shape[0]))
    img_rot = img_rot.astype(np.float32) / 255
    return img_rot

            

class MOSSETracker_Task2:

    def __init__(self, sigma=100, learning_rate=0.125, num_pretrain=128, rotate=False):
        self.template = None
        self.last_response = None
        self.region = None
        self.region_shape = None
        self.region_center = None
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.num_pretrain = num_pretrain
        self.rotate = rotate
        # Variables for MOSSE filter training
        self.Ai = None
        self.Bi = None
        self.G = None

    def get_region(self):
        return copy(self.region)

    def _get_gauss_response(self, image):
        height, width = image.shape
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        center_x = self.region.xpos + self.region_center[1]
        center_y = self.region.ypos + self.region_center[0]
        dist = (np.square(xx - center_x) + np.square(yy - center_y)) / (2 * self.sigma)
        response = np.exp(-dist)
        return response / np.sum(response)

    def _pre_training(self, init_frame, G):
        h, w = G.shape
        fi = cv2.resize(init_frame, (w, h))
        fi = pre_process(fi)
        Ai = G * np.conjugate(fft2(fi))
        Bi = fft2(fi) * np.conjugate(fft2(fi))
        for _ in range(self.num_pretrain):
            if self.rotate:
                fi_warp = random_warp(init_frame)
            else:
                fi_warp = init_frame
            fi_warp = cv2.resize(fi_warp, (w, h))
            fi_warp = pre_process(fi_warp)
            Fi = fft2(fi_warp)
            Ai += G * np.conjugate(Fi)
            Bi += Fi * np.conjugate(Fi)
        return Ai, Bi

    def start(self, image, region):
        num_channel = image.shape[-1]
        
        self.region = copy(region)
        self.region_shape = (region.height, region.width)
        self.region_center = (region.height // 2, region.width // 2)
        self.G = []
        self.Ai = []
        self.Bi = []
        for c in range(num_channel):
            response_map = self._get_gauss_response(image[:, :, c])
            g = crop_patch(response_map, region)
            fi = crop_patch(image[:, :, c], region)
            self.G.append(fft2(g))
            Ai, Bi = self._pre_training(fi, self.G[-1])
            self.Ai.append(Ai)
            self.Bi.append(Bi)
            

    def detect(self, image):
        num_channel = image.shape[-1]
        
        gi_channels = []
        for c in range(num_channel):

            Hi_current = self.Ai[c] / (self.Bi[c] + 1e-6)

            # Extract region for detection
            fi = crop_patch(image[:,:,c], self.region)
            fi = cv2.resize(fi, (self.region.width, self.region.height))
            fi = pre_process(fi)
            
            Gi_current = Hi_current * fft2(fi)
            gi_current = ifft2(Gi_current).real
            gi_channels.append(gi_current)

        gi_combined = sum(gi_channels) / num_channel
        
        #print(gi_combined)
        # Find max response
        max_pos = np.where(gi_combined == np.max(gi_combined))
        dy = int(np.mean(max_pos[0]) - gi_combined.shape[0] / 2)
        dx = int(np.mean(max_pos[1]) - gi_combined.shape[1] / 2)
        
        self.region.xpos += dx
        self.region.ypos += dy
        
        return self.get_region()


    def update(self, image):
        # Online update of the filters
        num_channel = image.shape[-1]
        region = self.region
        lr = self.learning_rate

        for c in range(num_channel):
            fi = crop_patch(image[:,:,c], region)
            fi = cv2.resize(fi, (region.width, region.height))
            fi = pre_process(fi)
            Fi = fft2(fi)
               
            #print("self.G[c]", self.G[c].shape)
            #print("fi", Fi.shape)  
            self.Ai[c] = lr * (self.G[c] * np.conjugate(Fi)) + (1 - lr) * self.Ai[c]
            self.Bi[c] = lr * (Fi * np.conjugate(Fi)) + (1 - lr) * self.Bi[c]





class MOSSETracker_Task3(MOSSETracker_Task2):

    def __init__(self,
                 sigma=100,
                 learning_rate=0.125,
                 num_pretrain=128,
                 rotate=False,
                 backbone: DeepFeatureExtractor = DeepFeatureExtractor(network_type='resnet34'),
                 used_resnet_layer: int = 1
                ):
        super().__init__(
            sigma=100,
            learning_rate=0.125,
            num_pretrain=128,
            rotate=False,
        )
        self.__backbone = backbone
        self._used_resnet_layer = used_resnet_layer
        self.scaling_height = None
        self.scaling_width = None

    def compute_deepfeatures(self, image):
        original_size = image.shape[:2]
        features = self.__backbone(image)[self._used_resnet_layer][0].unsqueeze(0)
        features = interpolate(features, size=original_size, mode='bilinear', align_corners=False)
        features = features[0].transpose(0,2).transpose(0,1)
        return features.cpu().numpy()


    def start(self, image, region):
        features = self.compute_deepfeatures(image)
        return super().start(features, region)

        
    def detect(self, image):
        return super().detect(self.compute_deepfeatures(image))
    
    def update(self, image):
        return super().update(self.compute_deepfeatures(image))











class MOSSETracker_Task1:

    def __init__(self, sigma=100, learning_rate=0.125, num_pretrain=128, rotate=False):
        self.template = None
        self.last_response = None
        self.region = None
        self.region_shape = None
        self.region_center = None
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.num_pretrain = num_pretrain
        self.rotate = rotate
        # Variables for MOSSE filter training
        self.Ai = None
        self.Bi = None
        self.G = None

    def get_region(self):
        return copy(self.region)

    def _get_gauss_response(self, image):
        height, width = image.shape
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        center_x = self.region.xpos + self.region_center[1]
        center_y = self.region.ypos + self.region_center[0]
        dist = (np.square(xx - center_x) + np.square(yy - center_y)) / (2 * self.sigma)
        response = np.exp(-dist)
        return response / np.sum(response)

    def _pre_training(self, init_frame, G):
        h, w = G.shape
        fi = cv2.resize(init_frame, (w, h))
        fi = pre_process(fi)
        Ai = G * np.conjugate(fft2(fi))
        Bi = fft2(fi) * np.conjugate(fft2(fi))
        for _ in range(self.num_pretrain):
            if self.rotate:
                fi_warp = random_warp(init_frame)
            else:
                fi_warp = init_frame
            fi_warp = cv2.resize(fi_warp, (w, h))
            fi_warp = pre_process(fi_warp)
            Fi = fft2(fi_warp)
            Ai += G * np.conjugate(Fi)
            Bi += Fi * np.conjugate(Fi)
        return Ai, Bi

    def start(self, image, region):
        #assert len(image.shape) == 2, "MOSSE is only defined for grayscale images"
        image = np.sum(image, 2) / 3
        self.region = copy(region)
        self.region_shape = (region.height, region.width)
        self.region_center = (region.height // 2, region.width // 2)
        
        # Create initial Gaussian response
        response_map = self._get_gauss_response(image)
        g = crop_patch(response_map, region)
        fi = crop_patch(image, region)
        self.G = fft2(g)
        self.Ai, self.Bi = self._pre_training(fi, self.G)

    def detect(self, image):
        #assert len(image.shape) == 2, "MOSSE is only defined for grayscale images"
        image = np.sum(image, 2) / 3

        Hi = self.Ai / (self.Bi + 1e-6)
        # Extract region for detection
        fi = crop_patch(image, self.region)
        fi = cv2.resize(fi, (self.region.width, self.region.height))
        fi = pre_process(fi)
        Gi = Hi * fft2(fi)
        gi = ifft2(Gi).real
        # Find max response
        max_pos = np.where(gi == np.max(gi))
        dy = int(np.mean(max_pos[0]) - gi.shape[0] / 2)
        dx = int(np.mean(max_pos[1]) - gi.shape[1] / 2)
        self.region.xpos += dx
        self.region.ypos += dy
        return self.get_region()

    def update(self, image):
        #assert len(image.shape) == 2, "MOSSE is only defined for grayscale images"
        image = np.sum(image, 2) / 3

        # Online update of the filters
        region = self.region
        fi = crop_patch(image, self.region)
        fi = cv2.resize(fi, (region.width, region.height))
        fi = pre_process(fi)
        Fi = fft2(fi)
        lr = self.learning_rate
        self.Ai = lr * (self.G * np.conjugate(Fi)) + (1 - lr) * self.Ai
        self.Bi = lr * (Fi * np.conjugate(Fi)) + (1 - lr) * self.Bi

class NCCTracker:

    def __init__(self, learning_rate=0.1):
        self.template = None
        self.last_response = None
        self.region = None
        self.region_shape = None
        self.region_center = None
        self.learning_rate = learning_rate

    def get_region(self):
        return copy(self.region)

    def get_normalized_patch(self, image):
        region = self.region
        patch = crop_patch(image, region)
        patch = patch / 255
        patch = patch - np.mean(patch)
        patch = patch / np.std(patch)
        return patch

    def start(self, image, region):
        assert len(image.shape) == 2, "NCC is only defined for grayscale images"
        self.region = copy(region)
        self.region_shape = (region.height, region.width)
        self.region_center = (region.height // 2, region.width // 2)
        patch = self.get_normalized_patch(image)
        self.template = fft2(patch)

    def detect(self, image):
        assert len(image.shape) == 2, "NCC is only defined for grayscale images"
        patch = self.get_normalized_patch(image)
        
        patchf = fft2(patch)

        responsef = self.template * np.conj(patchf)
        response = ifft2(responsef).real

        r, c = np.unravel_index(np.argmax(response), response.shape)

        # Keep for visualisation
        self.last_response = response

        r_offset = np.mod(r + self.region_center[0], self.region.height) - self.region_center[0]
        c_offset = np.mod(c + self.region_center[1], self.region.width) - self.region_center[1]

        self.region.xpos += c_offset
        self.region.ypos += r_offset

        return self.get_region()

    def update(self, image):
        assert len(image.shape) == 2, "NCC is only defined for grayscale images"
        patch = self.get_normalized_patch(image)
        patchf = fft2(patch)
        lr = self.learning_rate
        self.template = self.template * (1 - lr) + patchf * lr
