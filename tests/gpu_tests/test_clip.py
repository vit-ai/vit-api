import requests
import sys
import os
import json
import numpy as np


def call_endpoint(image_path, food_list):
    # Check if the file exists
    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' does not exist.")
        return

    # Define the endpoint URL
    endpoint_url = " https://liam-mckenna04--clip-inference-dev.modal.run" 

    # Open the image file in binary mode
    with open(image_path, "rb") as image_file:
        # Prepare the multipart request
        file_name = os.path.basename(image_path)

        files = {'uploadfile': (file_name, image_file, 'image/jpeg'), 'label_list': ('items_list', json.dumps(food_list), 'application/json')}
        response = requests.post(endpoint_url, files=files)
        # Check if the response status is 200 OK
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.reason}")
        content = json.loads(response.content)

    return content


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
    content = call_endpoint(img_path, food_list)
    print(content)