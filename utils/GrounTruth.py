from skimage.draw import polygon2mask
import numpy as np

def polygon2Mask(polygon:list):
    """Convert polgon to np.array"""
    polygon_point_list=[]
    for i in range(0,len(polygon),2):
        polygon_point_list.append([polygon[i+1], polygon[i]])
    
    polygon_point_list_arr= np.array(polygon_point_list)

    mask= polygon2mask((1080,1920),polygon_point_list_arr)

    return mask


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
