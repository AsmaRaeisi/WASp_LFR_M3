from typing import Literal, NamedTuple, Optional, TypedDict
import numpy as np
import cv2
from PIL import Image
import os
from utils import pre_process, random_warp

from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, resnet
from torch import nn
import torch

"""
This module implements the basic correlation filter based tracking algorithm -- MOSSE

Date: 2018-05-28

"""

RESNET_TYPE = Literal["ResNet18_Weights", "ResNet34_Weights",
                      "ResNet50_Weights", "ResNet101_Weights"]

class ResNetFeatureExtractor(nn.Module):

    def __init__(self, pretrained: bool = False,
                 resnet_version: RESNET_TYPE = "ResNet34_Weights") -> None:
        super().__init__()


        if resnet_version == "ResNet18_Weights":
            selected_meta = ResNet18_Weights
            selected_class = resnet.resnet18

        elif resnet_version == "ResNet34_Weights":
            selected_meta = ResNet34_Weights
            selected_class = resnet.resnet34

        elif resnet_version == "ResNet50_Weights":
            selected_meta = ResNet50_Weights
            selected_class = resnet.resnet50

        elif resnet_version == "ResNet101_Weights":
            selected_meta = ResNet101_Weights
            selected_class = resnet.resnet101
        else:
            raise Exception(f'{resnet_version} is not implemented yet.')

        weights = selected_meta.DEFAULT if pretrained else None

        backbone = selected_class(weights=weights)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).reshape(3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).reshape(3,1,1))
        if torch.cuda.is_available():
            self.to("cuda:0")
            self.device = "cuda:0"
        else:
            self.device = "cpu"

    def transforms(self, img):
        x = (torch.tensor(img).permute(2,0,1)[None].float().to(self.device) / 255.0 - self.mean) / self.std
        return x


    def _reshape_output(self, hidden_features: torch.Tensor):
        """This function reshapes the hidden features
        s.t. they are in line with the remaining Mosse implementation.

        Args:
            hidden_features : The hidden features of a single feature
        """
        return hidden_features[0].transpose(0,2).transpose(0,1)
        
    def forward(self, img):

        x = self.transforms(img)
        # x = torch.unsqueeze(x, 0)
        x = x.to(self.device)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return  self._reshape_output(x1),\
                self._reshape_output(x2),\
                self._reshape_output(x3),\
                self._reshape_output(x4)

# used for linear mapping...
def linear_mapping(img):
    return (img - img.min()) / (img.max() - img.min() + 1e-6)

class Mosse:
    def __init__(self, args, img_path, backbone: nn.Module):
        # get arguments..
        self.args = args
        self.img_path = img_path
        # get the img lists...
        self.frame_lists = self._get_img_lists(self.img_path)
        self.frame_lists.sort()
        self.backbone = backbone
    
    # start to do the object tracking...
    def start_tracking(self, used_resnet_layer: int = 2):
        
        # get the image of the first frame... 
       
        init_img = cv2.imread(self.frame_lists[0], cv2.IMREAD_UNCHANGED)  # Support multi-channel images
        # init_img = Image.open(self.frame_lists[0])
        init_img = init_img.astype(np.float32)

        features = self.backbone(init_img)[used_resnet_layer].cpu().numpy()

        num_channels = features.shape[2]  # Get the number of channels

        # TODO: change
        # get the init ground truth.. [x, y, width, height]
        init_gt = cv2.selectROI('demo', init_img.astype(np.uint8), False, False)
        # init_gt = (238, 133, 122, 165)
        init_gt = np.array(init_gt).astype(np.int64)

        scale_h = np.floor(init_img.shape[0] / features.shape[0])
        scale_w = np.floor(init_img.shape[1] /features.shape[1])
        scaled_bbox = [
            int(init_gt[0] / scale_w), # Scale x
            int(init_gt[1] / scale_h), # Scale y
            int(init_gt[2] / scale_w), # Scale width
            int(init_gt[3] / scale_h), # Scale height
        ]

        
        # start to draw the gaussian response...
        response_maps = []

        # Compute Gaussian response for each channel
        for channel in range(num_channels):
                    response_maps.append(self._get_gauss_response(features[:, :, channel], scaled_bbox))

        # start to create the training set ...
    

        filters = [{'A': None, 'B': None} for _ in range(num_channels)]
        for channel in range(num_channels):
            g = response_maps[channel][scaled_bbox[1]:scaled_bbox[1] + scaled_bbox[3], scaled_bbox[0]:scaled_bbox[0] + scaled_bbox[2]]
            fi = features[scaled_bbox[1]:scaled_bbox[1] + scaled_bbox[3], scaled_bbox[0]:scaled_bbox[0] + scaled_bbox[2], channel]
            G = np.fft.fft2(g)
            Ai, Bi = self._pre_training(fi, G)
            filters[channel]['A'], filters[channel]['B'] = Ai, Bi
            
        
        
        # start the tracking...

        pos = init_gt.copy()
        for idx, frame_path in enumerate(self.frame_lists):

            current_frame = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            current_features = self.backbone(current_frame)[used_resnet_layer].cpu().numpy()
            if idx == 0:
                x_original = scaled_bbox[0]
                y_original = scaled_bbox[1]
                w_original = scaled_bbox[2]
                h_original = scaled_bbox[3]
                clip_pos = np.array([x_original, y_original, x_original + w_original, y_original + h_original]).astype(np.int64)
            else:
                responses = []
                for channel in range(num_channels):
                    Ai, Bi = filters[channel]['A'], filters[channel]['B']
                    Hi = Ai / Bi
                    fi = current_features[clip_pos[1]:clip_pos[3], clip_pos[0]:clip_pos[2], channel]
                    fi = pre_process(cv2.resize(fi, (scaled_bbox[2], scaled_bbox[3])))
            
        
                    Gi = Hi * np.fft.fft2(fi)
                    gi = linear_mapping(np.fft.ifft2(Gi))
                    responses.append(gi)

                # Combine responses (average across channels)
                combined_response = sum(responses) / num_channels

                # Find max response position
                max_value = np.max(combined_response)
                max_pos = np.where(combined_response == max_value)
                dy = int(np.mean(max_pos[0]) - combined_response.shape[0] / 2)
                dx = int(np.mean(max_pos[1]) - combined_response.shape[1] / 2)

                # Update position
                scaled_bbox[0] += dx
                scaled_bbox[1] += dy
                clip_pos[0] = np.clip(scaled_bbox[0], 0, current_features.shape[1])
                clip_pos[1] = np.clip(scaled_bbox[1], 0, current_features.shape[0])
                clip_pos[2] = np.clip(scaled_bbox[0] + scaled_bbox[2], 0, current_features.shape[1])
                clip_pos[3] = np.clip(scaled_bbox[1] + scaled_bbox[3], 0, current_features.shape[0])
                clip_pos = clip_pos.astype(np.int64)

                # Update filters for each channel
                for channel in range(num_channels):
                    fi = current_features[clip_pos[1]:clip_pos[3], clip_pos[0]:clip_pos[2], channel]
                    
                    fi = pre_process(cv2.resize(fi, (scaled_bbox[2], scaled_bbox[3])))
                    G = np.fft.fft2(fi)
                    filters[channel]['A'] = self.args.lr * (G * np.conjugate(G)) + (1 - self.args.lr) * filters[channel]['A']
                    filters[channel]['B'] = self.args.lr * (G * np.conjugate(G)) + (1 - self.args.lr) * filters[channel]['B']
            
            # visualize the tracking process...
            ##### color visualization       

            x = int(clip_pos[0] * scale_w)
            y = int(clip_pos[1] * scale_h)
            x2 = int(clip_pos[2] * scale_w)
            y2 = int(clip_pos[3] * scale_h)

            cv2.rectangle(current_frame, (x, y), (x2, y2), (255, 0, 0), 2)
            # cv needs the image to be uint8 to ignore scaling
            cv2.imshow('demo', current_frame.astype(np.uint8))
            cv2.waitKey(200)
            # if record... save the frames..
            if self.args.record:
                frame_path = 'record_frames/' + self.img_path.split('/')[1] + '/'
                if not os.path.exists(frame_path):
                    os.makedirs(frame_path)
                cv2.imwrite(frame_path + str(idx).zfill(5) + '.png', current_features)
                
                
    # pre train the filter on the first frame...
    def _pre_training(self, init_frame, G):
        height, width = G.shape
        fi = cv2.resize(init_frame, (width, height))
        # pre-process img..
        fi = pre_process(fi)

        Ai = G * np.conjugate(np.fft.fft2(fi))
        Bi = np.fft.fft2(init_frame) * np.conjugate(np.fft.fft2(init_frame))

        for _ in range(self.args.num_pretrain):
            if self.args.rotate:
                fi = pre_process(random_warp(init_frame))
            else:
                fi = pre_process(init_frame)

            Ai = Ai + G * np.conjugate(np.fft.fft2(fi))
            Bi = Bi + np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))
        
        Ai += 1e-6
        Bi += 1e-6
        return Ai, Bi

    # get the ground-truth gaussian reponse...
    def _get_gauss_response(self, img, gt):
        # get the shape of the image..
        height, width = img.shape
        # get the mesh grid...
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        # get the center of the object...
        center_x = gt[0] + 0.5 * gt[2]
        center_y = gt[1] + 0.5 * gt[3]
        # cal the distance...
        dist = (np.square(xx - center_x) + np.square(yy - center_y)) / (2 * self.args.sigma)
        # get the response map...
        response = np.exp(-dist)
        # normalize...
        response = linear_mapping(response)
        return response

    # it will extract the image list 
    def _get_img_lists(self, img_path):
        frame_list = []
        for frame in os.listdir(img_path):
            if os.path.splitext(frame)[1] == '.jpg':
                frame_list.append(os.path.join(img_path, frame)) 
        return frame_list
    
    # it will get the first ground truth of the video..
    def _get_init_ground_truth(self, img_path):
        gt_path = os.path.join(img_path, 'groundtruth.txt')
        with open(gt_path, 'r') as f:
            # just read the first frame...
            line = f.readline()
            gt_pos = line.split(',')

        return [float(element) for element in gt_pos]

# class MosseArgDict(TypedDict):
#     lr: float
#     sigma: float
#     num_pretrain: int
#     rotate: bool
#     record: bool

class MosseArgDict(NamedTuple):
    lr: float
    sigma: float
    num_pretrain: int
    rotate: bool
    record: bool

if __name__ == '__main__':
    backbone = ResNetFeatureExtractor()
    mosse_args = MosseArgDict(
        lr=0.125,
        sigma=100.0,
        num_pretrain=128,
        rotate=False,
        record=False,
    )

    img_path = './TrackingProject/Mini-OTB/Surfer/img/'
    mosse = Mosse(args=mosse_args, img_path=img_path, backbone=backbone)
    mosse.start_tracking()
