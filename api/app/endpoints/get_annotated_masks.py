import sys
import pandas as pd




def split_masks(pil_image, yolo=False): 
    sys.path.append("endpoints/util")
    import request_util as ru
    import image_util as iu
    print('split_masks')
    modal_endpoint_object_detection = "dino-inference"
    if yolo:
        modal_endpoint_object_detection = "yolo-inference"
    # Object detection
    print('calling object detection')
    boxes, logits = ru.call_endpoint(pil_image, modal_endpoint_object_detection)
    print('done calling object detection')
    boxes = iu.filter_boxes(boxes)
    cropped_images = []
    for box, logit in zip(boxes, logits): 

        cropped_image = pil_image.crop(box)
        cropped_images.append({"image": cropped_image, "box": box, "logit": logit})
    return cropped_images
#*
# Annotated masks are returned as a list of dictionaries. Each dictionary contains the following keys:


# *#
def get_annotated_masks(pil_image, subsegment=False):
    sys.path.append("endpoints/util")
    import request_util as ru
    import image_util as iu
    import numpy as np
    print('get_annotated_masks')
    modal_endpoint_semantic_segmentation = "semantic-sam-inference"
    modal_endpoint_classification = "clip-inference"
 
    print(pil_image.size)
    #'segmentation', 'area', 'bbox', 'predicted_iou', 'point_coords', 'stability_score', 'crop_box', and 
        # Assuming you would call your semantic segmentation endpoint similarly
    masks: list[dict] = ru.call_endpoint(pil_image, modal_endpoint_semantic_segmentation, levels=[1], test=False)
 
    
    print('done calling segmentation')

    print(len(masks))
    for mask in masks:
        print(np.array(mask['segmentation']).shape)
    # Clip food classification
    food_list = pd.read_csv("endpoints/util/food_data.csv")['description'].tolist()

    # Create cutouts/crops based on segments 
    # mask_crops = iu.crop_from_mask(pil_image, masks)
    # # mask_segments = iu.segmented_from_mask(pil_image, masks)
    # for crop, mask in zip(mask_crops, masks):
    #     mask['labels'] = ru.call_endpoint(crop, modal_endpoint_classification, label_list=food_list)
        
    
    # Return annotated masks
    return masks

