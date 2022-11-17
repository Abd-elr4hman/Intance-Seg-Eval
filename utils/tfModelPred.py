from object_detection.utils import ops as utils_ops
import cv2
import tensorflow as tf
import numpy as np

def image_file_to_tensor(path):
    cv_img = cv2.imread(path,1).astype('uint8')
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    image_arr = cv2.resize(cv_img, (960, 512))   
    return image_arr

def image_tensor_to_batched_tensor(image_arr):
    batched_tensor=np.expand_dims(image_arr, 0)
    return batched_tensor

def reframe_masks(detections, w, h):
    """transforms detection masks from 33x33 to image size"""
    if 'detection_masks' in detections:
        detection_masks_reframed= utils_ops.reframe_box_masks_to_image_masks(detections['detection_masks'][0],
        detections['detection_boxes'][0], w, h)
        detection_masks_reframed= tf.cast(detection_masks_reframed> 0.5,tf.int32)
        masks = detection_masks_reframed.numpy()

    return masks

def postprocess_detections(detections:dict, image_arr:np.array):
    """Postprocess detections dict into result np arrays"""
    boxes= detections['detection_boxes'].numpy()[0]
    classes= detections['detection_classes'].numpy()[0].astype(int)
    scores= detections['detection_scores'].numpy()[0]

    w,h = image_arr.shape[0],image_arr.shape[1]
    masks= reframe_masks(detections, w, h)
    return boxes, classes, scores, masks

def distil_detections(boxes, classes, scores, masks, min_score_thresh=0.5):
    """returns only detections with score above min_score_thresh"""
    distilled_boxes= []
    distilled_classes=[]
    distilled_scores=[]
    distilled_masks=[]

    for i in range(boxes.shape[0]):
        if scores[i] > min_score_thresh:
            distilled_boxes.append(boxes[i])
            distilled_classes.append(classes[i])
            distilled_scores.append(scores[i])
            distilled_masks.append(masks[i])

    return np.array(distilled_boxes), np.array(distilled_classes), np.array(distilled_scores), np.array(distilled_masks)

def create_detections_array(distilled_boxes, distilled_classes, distilled_scores, distilled_masks):
    """transforms distilled_detections into the form needed by process_batch()

    Args:
        distilled_boxes, distilled_classes, distilled_scores, distilled_masks
    Returns:
        (Array[N, 6]), y1, x1, y2, x2, conf, class
    """
    detections_list=[]
    for i in range(distilled_boxes.shape[0]):
        detections_instance=[]

        # add bbox at positions  0:3
        for item in distilled_boxes[i]:
            detections_instance.append(item)

        # add conf at position 4
        detections_instance.append(distilled_scores[i])

        # add class at position 5
        detections_instance.append(distilled_classes[i])

        detections_list.append(detections_instance)
    
    return np.array(detections_list)


def get_detection_and_pred_masks(detections, image_arr):
    """takes model output detections and performs postprocessing

    Args:
        detections: tf.maskRCNN output
        image_arr: inference image
    Returns:
        detec: (Array[N, 6]), y1, x1, y2, x2, conf, class.
        distilled_masks, distilled_boxes, distilled_classes, distilled_scores
    """
    # post process detections
    boxes, classes, scores, masks= postprocess_detections(detections, image_arr)

    # distill detections
    distilled_boxes, distilled_classes, distilled_scores, distilled_masks= distil_detections(boxes, classes, scores, masks, min_score_thresh=0.5)

    # create detec array
    detec= create_detections_array(distilled_boxes, distilled_classes, distilled_scores, distilled_masks)

    return detec, distilled_masks, distilled_boxes, distilled_classes, distilled_scores