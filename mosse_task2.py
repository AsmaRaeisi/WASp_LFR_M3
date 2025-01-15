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
        self.ground_truths = self._load_ground_truths()
        self.total_iou = 0
        self.total_precision = 0
        self.num_frames = len(self.frame_lists)        
    
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
            
            current_gt = self.ground_truths[idx]
            current_gt = np.array(current_gt).astype(np.int64)      
            
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
                    Fi = np.fft.fft2(fi)
                    filters[channel]['A'] = self.args.lr * (G * np.conjugate(Fi)) + (1 - self.args.lr) * filters[channel]['A']
                    filters[channel]['B'] = self.args.lr * (Fi * np.conjugate(Fi)) + (1 - self.args.lr) * filters[channel]['B']

            
            
            
            
            # visualize the tracking process...
            ##### color visualization       
            cv2.rectangle(current_frame, (pos[0], pos[1]), (pos[0]+pos[2], pos[1]+pos[3]), (255, 0, 0), 2)
            
            # Draw the ground truth rectangle (green)
            cv2.rectangle(current_frame, (current_gt[0], current_gt[1]), (current_gt[0]+current_gt[2], current_gt[1]+current_gt[3]), (0, 255, 0), 2) 
            
            cv2.imshow('demo', current_frame)
            cv2.waitKey(100)
            # if record... save the frames..
            if self.args.record:
                frame_path = 'record_frames/' + self.img_path.split('/')[1] + '/'
                if not os.path.exists(frame_path):
                    os.makedirs(frame_path)
                cv2.imwrite(frame_path + str(idx).zfill(5) + '.png', current_frame)
                
            self.total_iou += self._calculate_iou(pos, current_gt)
            self.total_precision += self._calculate_precision(pos, current_gt)

        avg_iou = self.total_iou / self.num_frames
        avg_precision = self.total_precision / self.num_frames
        print(f'Average IOU: {avg_iou:.4f}')
        print(f'Average Precision: {avg_precision:.4f}')                
                
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
    
    def _load_ground_truths(self):
        gt_path = os.path.join(self.img_path, '../groundtruth.txt')
        ground_truths = []
        with open(gt_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                ground_truths.append([int(float(p)) for p in parts])
        return ground_truths
        
    def _calculate_iou(self, pred, gt):
        inter_xmin = max(pred[0], gt[0])
        inter_ymin = max(pred[1], gt[1])
        inter_xmax = min(pred[0] + pred[2], gt[0] + gt[2])
        inter_ymax = min(pred[1] + pred[3], gt[1] + gt[3])
        inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
        pred_area = pred[2] * pred[3]
        gt_area = gt[2] * gt[3]
        union_area = pred_area + gt_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def _calculate_precision(self, pred, gt):
        pred_center = [pred[0] + pred[2] / 2, pred[1] + pred[3] / 2]
        gt_center = [gt[0] + gt[2] / 2, gt[1] + gt[3] / 2]
        distance = np.sqrt((pred_center[0] - gt_center[0]) ** 2 + (pred_center[1] - gt_center[1]) ** 2)
        return 1 if distance <= max(gt[2], gt[3]) * 0.5 else 0
        
