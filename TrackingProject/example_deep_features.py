"""
Created by: Gustav Häger
Updated by: Johan Edstedt (2021)
"""


import argparse
from tqdm import tqdm
from cvl.dataset import OnlineTrackingBenchmark
from cvl.features_resnet import DeepFeatureExtractor



if __name__ == "__main__":
    parser = argparse.ArgumentParser('Args for the tracker')
    parser.add_argument('--sequences',nargs="+",default=[3, 4, 5],type=int)
    parser.add_argument('--dataset_path',type=str,default="/courses/TSBB19/otb_mini")
    args = parser.parse_args()

    dataset_path,sequences = args.dataset_path,args.sequences

    dataset = OnlineTrackingBenchmark(dataset_path)
    feature_extractor = DeepFeatureExtractor()

    for sequence_idx in tqdm(sequences):
        a_seq = dataset[sequence_idx]
        
        for frame_idx, frame in tqdm(enumerate(a_seq), leave=False):
            image_color = frame['image']
            deep_features = feature_extractor(image_color)
            