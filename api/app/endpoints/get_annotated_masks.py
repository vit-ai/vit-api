import sys


#*
# Annotated masks are returned as a list of dictionaries. Each dictionary contains the following keys:


# *#
def get_annotated_masks(pil_image, subsegment=False, yolo=False):
    sys.path.append("endpoints/util")
    import request_util as ru
    import image_util as iu
    print('get_annotated_masks')
    modal_endpoint_object_detection = "dino-inference"
    modal_endpoint_semantic_segmentation = "semantic-sam-inference"
    if yolo:
        modal_endpoint_object_detection = "yolo-inference"
    # Object detection
    boxes, logits = ru.call_endpoint(pil_image, modal_endpoint_object_detection)

    #do logic with logits here

    # filter out boxes that are within larger boxes 
    # (this is how we currently avoid double counting, may need to implement a better method in the future)
    boxes = iu.filter_boxes(boxes)

    # Semantic segmentation
    segmented_images = []
    for box in boxes:
        # Crop the original image using the bounding box coordinates
        cropped_image = pil_image.crop(box)
        
        # Assuming you would call your semantic segmentation endpoint similarly
        segmentation_result = ru.call_endpoint(cropped_image, modal_endpoint_semantic_segmentation, levels=[1], test=False)
        segmented_images.append(segmentation_result)
 
    

    print(segmented_images)
    # Clip food classification

    # Subsegmentation


    # Classify Subsegments

    # Return annotated masks
