# Intance-Seg-Eval
Evaluation code for instance Segmentation task.

# Contents
- [Evaluate](https://github.com/Abd-elr4hman/Intance-Seg-Eval/blob/main/Evaluate.ipynb): A notebook to Evaluate tf MaskRCNN trained on Road Distress dataset, the notebook downloads both the dataset in COCO format and the tf saved_model file first, then performs evaluation. 

# To use
1. Go through the installation process of TF OD API [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md).
2. Install dependancies in requirements.txt

# Acknowledgement
This repo uses code files from [YOLOv5](https://github.com/ultralytics/yolov5) and is based on their segmentation evaluation.
also borrows heavely from the following repos:
- [skimage](https://github.com/scikit-image/scikit-image/blob/main/skimage/draw/_polygon2mask.py)
- [Object-Detection-Metrics](https://github.com/rafaelpadilla/Object-Detection-Metrics)
- [object_detection_confusion_matrix](https://github.com/kaanakan/object_detection_confusion_matrix)
