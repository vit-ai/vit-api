import modal 
from fastapi import UploadFile


stub = modal.Stub('dino-inference')

image_dock = modal.Image.from_dockerfile('/Users/liammckenna/vit-ai/api/vit-api/gpu_api/Dockerfile').pip_install('groundingdino-py')

#todo: make sure this runs on GPU not CPU!!!!!!!!!!!!!
@stub.function(image=image_dock, mounts=[modal.Mount.from_local_dir("gpu_api/weights/grounding_dino", remote_path="/weights/")], gpu='T4')
@modal.web_endpoint(label='dino-inference', method="POST")
def grounding_dino_inference(uploadfile: UploadFile):
    import sys
    from groundingdino.util.inference import load_model, load_image, predict, annotate
    import groundingdino.datasets.transforms as T
    import cv2
    import numpy as np
    import os
    from PIL import Image
    from fastapi.responses import Response
    import json
    import torch

    def box_convert(boxes, image):
        """
        Convert a list of percent cxcywh tensors to a list of xyxy tensors.

        :param boxes: List of tensors with shape [c_x, c_y, w, h].
        :param image: Image tensor.
        :return: List of tensors with shape [x1, y1, x2, y2].
        """
        # Get image dimensions
        img_h, img_w = image.shape[:2]
        
        converted_boxes = []

        for box in boxes:
            # Extract the cxcywh values
            c_x, c_y, w, h = box

            # Convert percentages to actual pixel values
            w_pixel = w * img_w
            h_pixel = h * img_h
            c_x_pixel = c_x * img_w
            c_y_pixel = c_y * img_h

            # Compute top-left and bottom-right coordinates
            x1 = c_x_pixel - (w_pixel / 2)
            y1 = c_y_pixel - (h_pixel / 2)
            x2 = c_x_pixel + (w_pixel / 2)
            y2 = c_y_pixel + (h_pixel / 2)

            converted_boxes.append([x1.item(), y1.item(), x2.item(), y2.item()])

        return converted_boxes
    file_bytes = uploadfile.file.read()
    image_bgr = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    # image = np.array(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    image = image_bgr
    print(image)
    model = load_model("/usr/local/lib/python3.8/dist-packages/groundingdino/config/GroundingDINO_SwinT_OGC.py", "/weights/groundingdino_swint_ogc.pth")
    TEXT_PROMPT = "food . food-drink"
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_transformed, _ = transform(Image.fromarray(image), None)
    boxes, logits, phrases = predict(
    model=model,
    image=image_transformed,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD
    )
    new_boxes = box_convert(boxes, image)
    print(new_boxes)
    # annotated_frame = annotate(image_source=image, boxes=boxes, logits=logits, phrases=phrases)
    json_str = json.dumps([new_boxes, logits.tolist()])
    return Response(content=json_str) #should be just returning boxes/logits/etc. not the image itself
