import modal 
from fastapi import UploadFile

stub = modal.Stub('yolo-inference')

build_image = (modal.Image.from_dockerfile('/Users/liammckenna/vit-ai/api/vit-api/gpu_api/Dockerfile').pip_install('ultralytics')
         )

@stub.function(image=build_image, mounts=[modal.Mount.from_local_dir("/Users/liammckenna/vit-ai/api/vit-api/gpu_api/weights/yolo", remote_path="/weights/")], gpu='T4')
@modal.web_endpoint(label='yolo-inference', method="POST")
def yolo_inferenc(uploadfile: UploadFile):
    import torch 
    import cv2
    import io
    import numpy as np
    import ultralytics
    from fastapi.responses import Response
    import os 
    import json
    image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']
    video_extensions = ['.avi', '.mp4', '.mov', '.mkv', '.flv', '.wmv']

    # Initialize the YOLO model
    model = ultralytics.YOLO('/weights/best.pt')

    # Load uploaded file
    file_bytes = uploadfile.file.read()
    file_stream = io.BytesIO(file_bytes)

    # Determine if input is image or video based on extension
    file_ext = os.path.splitext(uploadfile.filename.lower())[1]

    if file_ext in image_extensions:
        # Read image using OpenCV
        image_bgr = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Inference
        results = model(image)[0]


        # Convert bounding boxes to JSON
        bbox_json = json.dumps([results.boxes.xyxy.tolist(), results.boxes.conf.tolist()])

        # Return JSON
        return Response(content=bbox_json) 

    elif file_ext in video_extensions:
        pass
    else:
        raise ValueError("Unsupported file format. Please upload a valid image or video.")

