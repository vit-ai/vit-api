from io import BytesIO
from PIL import Image
import requests
import json
import os 
def call_endpoint(pil_image, modal_endpoint_name,  / , **kwargs):
    # Define the endpoint URL
    endpoint_url = "https://liam-mckenna04--" + modal_endpoint_name + "-dev.modal.run" 

    # Convert the PIL image to a bytes-like object
    image_byte_array = BytesIO()
    pil_image.save(image_byte_array, format='JPEG')
    image_bytes = image_byte_array.getvalue()

    # Prepare the multipart request
    files = {'uploadfile': ('image.jpg', image_bytes, 'image/jpeg')}

    response = requests.post(endpoint_url, files=files, data=kwargs)
    # Check if the response status is 200 OK
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.reason}")
    content = json.loads(response.content)
    
    return content


def call_endpoint_with_path(image_path, modal_endpoint_name,  / , **kwargs):
    # Check if the file exists
    if not os.path.exists(image_path):
        raise(f"Error: The file '{image_path}' does not exist.")
        return "", ""

    # Define the endpoint URL
    endpoint_url = "https://liam-mckenna04--" + modal_endpoint_name + "-dev.modal.run"

    # Open the image file in binary mode
    with open(image_path, "rb") as image_file:
        # Prepare the multipart request
        file_name = os.path.basename(image_path)

        files = {'uploadfile': (file_name, image_file, 'image/jpeg')} 
        response = requests.post(endpoint_url, files=files, data=kwargs)
        # Check if the response status is 200 OK
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.reason}")

    # Parse the response
    return json.loads(response.content)
