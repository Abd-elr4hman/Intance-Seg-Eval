import numpy as np
import skimage.transform as st

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