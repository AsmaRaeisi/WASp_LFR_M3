import numpy as np
import cv2
import os
from utils import linear_mapping, pre_process, random_warp

"""
This module implements the basic correlation filter based tracking algorithm -- MOSSE

Date: 2018-05-28

"""

class mosse:
    def __init__(self, args, img_path):
        # get arguments..
        self.args = args
        self.img_path = img_path
        # get the img lists...
        self.frame_lists = self._get_img_lists(self.img_path)
        self.frame_lists.sort()
    
    # start to do the object tracking...
    def start_tracking(self):
        
        # get the image of the first frame... 
       
        init_img = cv2.imread(self.frame_lists[0], cv2.IMREAD_UNCHANGED)  # Support multi-channel images

        init_img = init_img.astype(np.float32)
        num_channels = init_img.shape[2]  # Get the number of channels


        # get the init ground truth.. [x, y, width, height]
        init_gt = cv2.selectROI('demo', init_img.astype(np.uint8), False, False)
        init_gt = np.array(init_gt).astype(np.int64)
        
        # start to draw the gaussian response...
        response_maps = []

        # Compute Gaussian response for each channel
        for channel in range(num_channels):
                    response_maps.append(self._get_gauss_response(init_img[:, :, channel], init_gt))





        # start to create the training set ...
    

        filters = [{'A': None, 'B': None} for _ in range(num_channels)]
        for channel in range(num_channels):
            g = response_maps[channel][init_gt[1]:init_gt[1] + init_gt[3], init_gt[0]:init_gt[0] + init_gt[2]]
            fi = init_img[init_gt[1]:init_gt[1] + init_gt[3], init_gt[0]:init_gt[0] + init_gt[2], channel]
            G = np.fft.fft2(g)
            Ai, Bi = self._pre_training(fi, G)
            filters[channel]['A'], filters[channel]['B'] = Ai, Bi
            
        
        
        
        # start the tracking...

        pos = init_gt.copy()
        for idx, frame_path in enumerate(self.frame_lists):
            current_frame = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            if idx == 0:
                clip_pos = np.array([pos[0], pos[1], pos[0] + pos[2], pos[1] + pos[3]]).astype(np.int64)
            else:
                responses = []
                for channel in range(num_channels):
                    Ai, Bi = filters[channel]['A'], filters[channel]['B']
                    Hi = Ai / Bi
                    fi = current_frame[clip_pos[1]:clip_pos[3], clip_pos[0]:clip_pos[2], channel]
                    fi = pre_process(cv2.resize(fi, (init_gt[2], init_gt[3])))
            
        
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
                pos[0] += dx
                pos[1] += dy
                clip_pos[0] = np.clip(pos[0], 0, current_frame.shape[1])
                clip_pos[1] = np.clip(pos[1], 0, current_frame.shape[0])
                clip_pos[2] = np.clip(pos[0] + pos[2], 0, current_frame.shape[1])
                clip_pos[3] = np.clip(pos[1] + pos[3], 0, current_frame.shape[0])
                clip_pos = clip_pos.astype(np.int64)

                # Update filters for each channel
                for channel in range(num_channels):
                    fi = current_frame[clip_pos[1]:clip_pos[3], clip_pos[0]:clip_pos[2], channel]
                    
                    fi = pre_process(cv2.resize(fi, (init_gt[2], init_gt[3])))
                    G = np.fft.fft2(fi)
                    filters[channel]['A'] = self.args.lr * (G * np.conjugate(G)) + (1 - self.args.lr) * filters[channel]['A']
                    filters[channel]['B'] = self.args.lr * (G * np.conjugate(G)) + (1 - self.args.lr) * filters[channel]['B']

            
            
            
            
            # visualize the tracking process...
            ##### color visualization       
            cv2.rectangle(current_frame, (pos[0], pos[1]), (pos[0]+pos[2], pos[1]+pos[3]), (255, 0, 0), 2)
            cv2.imshow('demo', current_frame)
            cv2.waitKey(100)
            # if record... save the frames..
            if self.args.record:
                frame_path = 'record_frames/' + self.img_path.split('/')[1] + '/'
                if not os.path.exists(frame_path):
                    os.makedirs(frame_path)
                cv2.imwrite(frame_path + str(idx).zfill(5) + '.png', current_frame)
                
                
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

