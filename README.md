# WASp_LFR_M3

reference:
https://github.com/TianhongDai/mosse-object-tracking/tree/master



**how to run the code:**
place the datasets in the below structure:
datasets/
    Basketball/img/0001.jpg ...
    biker/img/0001.jpg ...
    ...




replace demo and mosse.py with the one I shared here.
(remeber to rename mosse_task1 to mosse.py)



**run the below command:**
python demo.py --dataset_path "datasets/Basketball" --record 

then select a bounding box on the image and press the space
it will store the frames in a folder named record_frames
