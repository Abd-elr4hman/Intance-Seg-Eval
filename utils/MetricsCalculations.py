import numpy as np
from utils.metrics import ap_per_class

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