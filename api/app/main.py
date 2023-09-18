from fastapi import FastAPI, UploadFile
import sys 
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
    annotated_masks = get_annotated_masks.get_annotated_masks(pil_image)
    print(annotated_masks)
     