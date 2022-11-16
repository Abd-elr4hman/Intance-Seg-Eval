# Intance-Seg-Eval
Evaluation code for instance Segmentation task.

# Contents
- [Evaluate](https://github.com/Abd-elr4hman/Intance-Seg-Eval/blob/main/Evaluate.ipynb): A notebook to Evaluate tf MaskRCNN trained on Road Distress dataset, the notebook downloads both the dataset in COCO format and the tf saved_model file first, then performs evaluation. 

# Acknowledgement
This repo uses code files from [YOLOv5](https://github.com/ultralytics/yolov5) and is based on thier segmentation evaluation.
also borrows heavely from the following repos:
- [skimage](https://github.com/scikit-image/scikit-image/blob/main/skimage/draw/_polygon2mask.py)
- [Object-Detection-Metrics](https://github.com/rafaelpadilla/Object-Detection-Metrics)
- [object_detection_confusion_matrix](https://github.com/kaanakan/object_detection_confusion_matrix)
