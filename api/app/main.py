from fastapi import FastAPI, UploadFile, Response
import sys 
import json
sys.path.append("/Users/liammckenna/vit-ai/api/vit-api/api/app/endpoints")
import get_annotated_masks
from PIL import Image
app = FastAPI()


@app.get("/") # all logic should be here, isolate gpu heavy functions to gpu_api
async def root():
    return {"message": "Hello World"}

@app.post("/get_annotated_masks")
async def get_annotated_masks_endpoint(file: UploadFile):
    """
    get annotated masks of food image

    Args:
        file (UploadFile): image file
    
    Returns:
        json: json of annotated masks

    """


    pil_image = Image.open(file.file)
    detected_food_clusters = get_annotated_masks.split_masks(pil_image)
    if len(detected_food_clusters) == 0:
        return Response(content=json.dumps([]), media_type="application/json")
    
    for food in detected_food_clusters:
        annotated_masks = get_annotated_masks.get_annotated_masks(food['image'])
        food['masks'] = annotated_masks
        food['image'] = None
    
    return Response(content=json.dumps(detected_food_clusters), media_type="application/json")
     