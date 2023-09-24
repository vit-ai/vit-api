import numpy as np
from PIL import Image


def filter_boxes(boxes):
    """Remove boxes that are completely enveloped by larger boxes."""
    filtered_boxes = []
    for box in boxes:
        is_enveloped = False
        for other_box in boxes:
            if (other_box[0] <= box[0] and other_box[1] <= box[1] and
                    other_box[2] >= box[2] and other_box[3] >= box[3]) and box != other_box:
                is_enveloped = True
                break
        if not is_enveloped:
            filtered_boxes.append(box)
    return filtered_boxes


def crop_from_mask(image, masks):
    """
    Crop the image based on the bounding box present in the masks.
    
    Parameters:
    - image: PIL image object
    - masks: List of masks, each containing 'bbox' for bounding box
    
    Returns:
    - List of cropped PIL images
    """
    cropped_images = []
    
    for mask in masks:
        bbox = mask['bbox']
        cropped_image = image.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
        cropped_images.append(cropped_image)
        
    return cropped_images
import numpy as np
from PIL import Image

def segmented_from_mask(image, masks):
    """
    Segment the image based on the masks provided.
    
    Parameters:
    - image: PIL image object
    - masks: List of masks, each containing 'segmentation' data
    
    Returns:
    - List of segmented PIL images
    """
    # Convert the PIL image to a numpy array
    image_np = np.array(image)
    segmented_images = []
    
    for mask in masks:
        segmentation_np = np.array(mask['segmentation'])  # Convert the mask list to a numpy array
        # Ensure the segmentation mask is binary and has the same shape as the image
        print(segmentation_np.shape)
        print(image_np.shape[:2])
        if segmentation_np.shape == image_np.shape[:2]:
            # Get the segmented region using the mask
            segmented_region = image_np * segmentation_np[:, :, None]
            # Convert the segmented region to a PIL image
            segmented_image = Image.fromarray(segmented_region.astype('uint8'))
            segmented_images.append(segmented_image)
        else:
            print("Error: Mask dimensions don't match the image dimensions.")
        
    return segmented_images
