import numpy as np
from skimage.draw import polygon2mask
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops
import cv2
import tensorflow as tf
import skimage.transform as st
from utils.metrics import ap_per_class

############################################################## GT_utils ################################################################


class COCO():
    def __init__(self, data):
        self.data= data
        self.cocoJSON_images= self.data['images']
        self.cocoAnnotations= self.data['annotations']
        self.cocoCategories= self.data['categories']

    def get_annotaions(self):
        return self.cocoAnnotations

    def get_images(self):
        return self.cocoJSON_images

    def get_categories(self):
        return self.cocoCategories


    def get_images_IDS_filenames(self):
        """Takes coco annotaion images and returns lists of image IDS and filenames"""
        image_IDs=[]
        image_filenames=[]

        for image in self.cocoJSON_images:
            image_IDs.append(image['id'])
            image_filenames.append(image['file_name'])

        return image_IDs, image_filenames

    def retrieve_image_GT(self, image_ID:int):
        """retrieve ground truth instanve annotations for a single image"""
        needed_image_annotations=[]
        for annotation_instance in self.cocoAnnotations:
            if annotation_instance['image_id']==image_ID:
                needed_image_annotations.append(annotation_instance)

        return needed_image_annotations


'''
def get_images_IDS_filenames(coco_annotations_images:list):
    """Takes coco annotaion images and returns lists of image IDS and filenames"""
    image_IDs=[]
    image_filenames=[]

    for image in coco_annotations_images:
        image_IDs.append(image['id'])
        image_filenames.append(image['file_name'])

    return image_IDs, image_filenames
 
'''

def polygon2Mask(polygon:list):
    """Convert polgon to np.array"""
    polygon_point_list=[]
    for i in range(0,len(polygon),2):
        polygon_point_list.append([polygon[i+1], polygon[i]])
    
    polygon_point_list_arr= np.array(polygon_point_list)

    mask= polygon2mask((1080,1920),polygon_point_list_arr)

    return mask

'''
def retrieve_image_GT(image_ID:int, coco_annotations):
    """retrieve ground truth instanve annotations for a single image"""
    needed_image_annotations=[]
    for annotation_instance in coco_annotations:
        if annotation_instance['image_id']==image_ID:
            needed_image_annotations.append(annotation_instance)

    return needed_image_annotations
'''

# utils
def single_image_anno_to_masks(single_image_anno:list):
    """Transforms a single image segmentation annotations to masks.

    Args:
        single_image_anno:  a list of annotations for a single image.
    Returns:
        array[M, H, W], where M is the number of masks.
        
    """
    image_masks=[]
    for instance_anno in single_image_anno:
        polygon= instance_anno['segmentation'][0]
        instance_mask= polygon2Mask(polygon)
        image_masks.append(instance_mask)

    return np.array(image_masks)


def xywh_to_xyxy(bbox:list):
    return [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]

def xyxy_to_yxyx(bbox:list):
    return [bbox[1], bbox[0], bbox[3], bbox[2]]


def normalize_xyxy_bbox(bbox:list, w,h):
    return [bbox[0]/w, bbox[1]/h, bbox[2]/w, bbox[3]/h]


def single_image_anno_to_bboxLabels(single_image_anno:list, image_shape:tuple):
    """Retrieve bbox and class in a single array called bbox_Labels

    Args:
        single_image_anno:  a list of annotations for a single image.
    Returns:
        #array[M, 5], (class, x1, y1, x2, y2)
        array[M, 5], normalized(class, y1, x1, y2, x2)
    """
    w,h = image_shape[0],image_shape[1]

    image_bboxLabels=[]
    normalized_yxyx_bboxes=[]

    for instance_anno in single_image_anno:
        bbox= instance_anno['bbox']      #  [top left x position, top left y position, width, height].

        # xywh to xyxy
        xyxy_bbox= xywh_to_xyxy(bbox)

        # normalized bbox
        normalized_xyxy_bbox= normalize_xyxy_bbox(xyxy_bbox, w, h)

        # xyxy to yxyx
        normalized_yxyx_bbox= xyxy_to_yxyx(normalized_xyxy_bbox)

        class_= instance_anno['category_id']

        # append class_id to bbox list
        normalized_yxyx_bbox.insert(0,class_)

        # append bbox list to image_bboxLabels
        image_bboxLabels.append(normalized_yxyx_bbox)

        #normalized_yxyx_bboxes.append(normalized_yxyx_bbox)

    return np.array(image_bboxLabels)# , np.array(normalized_yxyx_bboxes)


def single_image_anno_Processing(single_image_anno:list, image_shape:tuple):
    # get GT_masks
    GT_masks= single_image_anno_to_masks(single_image_anno)

    # get bboxLabels
    bboxLabels= single_image_anno_to_bboxLabels(single_image_anno, image_shape)

    return GT_masks, bboxLabels


############################################################## PRED_utils ################################################################

def image_file_to_tensor(path):
    cv_img = cv2.imread(path,1).astype('uint8')
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    image_arr = cv2.resize(cv_img, (960, 512))   
    return image_arr

def image_arr_to_batched_tensor(image_arr):
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

############################################################## Post_Process ################################################################

def resize_masks(GT_masks:np.array, PRED_masks:np.array, shape:tuple):
    """resizes input masks to shape"""
    GT_masks = np.transpose(GT_masks, axes=[1,2,0])
    GT_masks= st.resize(GT_masks, shape, order=0, preserve_range=True, anti_aliasing=False)

    PRED_masks = np.transpose(PRED_masks, axes=[1,2,0])
    PRED_masks= st.resize(PRED_masks, shape, order=0, preserve_range=True, anti_aliasing=False)

    return GT_masks, PRED_masks


def reshape_masks(GT_masks, PRED_masks):
    """reshape masks to shape (N,n) where:
        N: number of mask instances.
        n: imageH x imageW"""
    GT_masks = np.transpose(GT_masks, axes=[2,0,1])
    PRED_masks = np.transpose(PRED_masks, axes=[2,0,1])

    GT_masks= GT_masks.reshape(GT_masks.shape[0], -1)
    PRED_masks= PRED_masks.reshape(PRED_masks.shape[0], -1)
    return GT_masks, PRED_masks


def undo_reshape(reshaped_GT_masks, reshaped_PRED_masks):
    """undo reshape_masks operation, used for testing"""
    GT_masks= reshaped_GT_masks.reshape(reshaped_GT_masks.shape[0], 166, 166)
    PRED_masks= reshaped_PRED_masks.reshape(reshaped_PRED_masks.shape[0], 166, 166)
    return GT_masks, PRED_masks

############################################################## Metrics Calculation ################################################################
def mask_iou(mask1, mask2, eps=1e-7):
    """
    mask1: [N, n] N means number of predicted objects
    mask2: [M, n] M means number of gt objects
    Note: n means image_w x image_h
    return: masks iou, [N, M]
    """
    intersection = np.dot(mask1, mask2.T) #.clip(0)
    union = (mask1.sum(1)[:, None] + mask2.sum(1)[None]) - intersection  # (area1 + area2) - intersection
    return intersection / (union + eps)


def box_iou(boxes1, boxes2):
    # https://github.com/kaanakan/object_detection_confusion_matrix/blob/master/confusion_matrix.py#L4
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) or (y1, x1, y2, x2) format.
    Arguments:
        boxes1 (Array[N, 4])
        boxes2 (Array[M, 4])
    Returns:
        iou (Array[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2

    This implementation is taken from the above link and changed so that it only uses numpy..
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(boxes1.T)
    area2 = box_area(boxes2.T)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    inter = np.prod(np.clip(rb - lt, a_min=0, a_max=None), 2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

def process_batch(detec, labels, iouv, pred_masks=None, gt_masks=None, masks=False):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), y1, x1, y2, x2, conf, class
        labels (array[M, 5]), class, y1, x1, y2, x2
        pred_masks (array[M, n1]), M number of predicted objects, n= image h x image w
        gt_masks (array[N, n1]), N number of GT objects, n= image h x image w
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    if masks:
        iou = mask_iou(gt_masks, pred_masks, eps=1e-7)
    else: #boxes
        iou = box_iou(labels[:, 1:], detec[:, :4])

    # correct matrix of size(num of detections, iou levels)
    correct = np.zeros((detec.shape[0], iouv.shape[0])).astype(bool)

    correct_class = labels[:, 0:1] == detec[:, 5]

    for i in range(len(iouv)):
        x = np.where((iou >= iouv[i]) & correct_class)
        if x[0].shape[0]:
            matches = np.concatenate((np.stack(x, 1), iou[x[0], x[1]][:, None]), 1) # [label, detect, iou]
            '''
            ex matches:
            array([[ 0.        ,  0.        ,  0.60869565],
            [ 1.        ,  4.        ,  0.6281407 ],
            [ 2.        ,  8.        ,  0.648     ],
            [ 3.        ,  7.        ,  0.92200557],
            [ 4.        ,  6.        ,  0.92555831],
            [ 8.        ,  1.        ,  0.6372549 ],
            [ 9.        ,  5.        ,  0.66666667],
            [10.        ,  3.        ,  0.71428571]])
            '''
            # if there is more than one match
            if x[0].shape[0] > 1:

                # sort based on score
                matches = matches[matches[:, 2].argsort()[::-1]]

                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True           # correct of size(num of detections, iou levels)
    return correct


def get_names_dict(categories:list):
    names=[]
    for category_fict in categories:
        names.append(category_fict['name'])

    names = dict(enumerate(names, start=1))
    return names

def ap_per_class_box_and_mask(
        tp_m,
        tp_b,
        conf,
        pred_cls,
        target_cls,
        plot=False,
        save_dir=".",
        names=(),):
    # https://github.com/ultralytics/yolov5/blob/master/utils/segment/metrics.py#L17    
    """
    Args:
        tp_b: tp of boxes.
        tp_m: tp of masks.
        other arguments see `func: ap_per_class`.
    """
    results_boxes = ap_per_class(tp_b,
                                 conf,
                                 pred_cls,
                                 target_cls,
                                 plot=plot,
                                 save_dir=save_dir,
                                 names=names,
                                 prefix="Box")[2:]
    results_masks = ap_per_class(tp_m,
                                 conf,
                                 pred_cls,
                                 target_cls,
                                 plot=plot,
                                 save_dir=save_dir,
                                 names=names,
                                 prefix="Mask")[2:]

    results = {
        "boxes": {
            "p": results_boxes[0],
            "r": results_boxes[1],
            "ap": results_boxes[3],
            "f1": results_boxes[2],
            "ap_class": results_boxes[4]},
        "masks": {
            "p": results_masks[0],
            "r": results_masks[1],
            "ap": results_masks[3],
            "f1": results_masks[2],
            "ap_class": results_masks[4]}}
    return results

