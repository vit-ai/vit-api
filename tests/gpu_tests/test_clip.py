import requests
import sys
import os
import json
import numpy as np
from PIL import Image
sys.path.append("/Users/liammckenna/vit-ai/api/vit-api/api/app/endpoints/util")
import request_util as ru

if __name__ == "__main__":
    # Ensure that the image path is provided
    if len(sys.argv) < 2:
        print("Usage: python script_name.py /path/to/image.jpg")
        sys.exit(1)

    # Extract the image path from the command-line arguments
    img_path = sys.argv[1]

    import pandas as pd 
    df = pd.read_csv("/Users/liammckenna/vit-ai/api/vit-api/gpu_api/endpoints/utils/food_data.csv")
    # get descriptions from df and put them in a list 
    food_list = df['description'].tolist()
    modal_endpoint_classification = "clip-inference"

    pil_image = Image.open(img_path)
    content = ru.call_endpoint(pil_image, modal_endpoint_classification, label_list=food_list)
    print(content)