import numpy as np
import skimage.io
import os
# EIGEN
import load_image

def LOAD_MASK(img_path):
    """Generate instance masks for an image.
    Returns:
    masks: A bool array of shape [height, width, instance count] with
        one mask per instance.
    class_ids: a 1D array of class IDs of the instance masks.
    """
    
    # Get mask directory from image path
    mask_dir = img_path

    # Read mask files from .png image
    mask = []
    for f in next(os.walk(mask_dir))[2]:
        if f.endswith(".png"):
            m = skimage.io.imread(os.path.join(mask_dir, f)).astype(np.bool)
            mask.append(m)
    mask = np.stack(mask, axis=-1)
    print(mask[0])
    print(mask.shape)
    # Return mask, and array of class IDs of each instance. Since we have
    # one class ID, we return an array of ones
    return mask, np.ones([mask.shape[-1]], dtype=np.int32)

if __name__ == '__main__':
    img_path = '/home/hangwu/CyMePro/data/annotations/trimaps'
    mask, mlist = LOAD_MASK(img_path)
    print(mlist)
    raws, cols = mask[0].shape
    for i in range(raws):
        for j in range(cols):
            if mask[0][i][j]:
                print(1)

