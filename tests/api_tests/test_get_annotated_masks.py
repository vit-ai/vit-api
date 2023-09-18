import requests
import os 







if __name__ == "__main__":
    image_path = "/Users/liammckenna/vit-ai/api/vit-api/tests/gpu_tests/assets/photos/84010.jpg"
    if not os.path.exists(image_path):
        raise(f"Error: The file '{image_path}' does not exist.")
    
    endpoint_url = "http://127.0.0.1:8000/get_annotated_masks"
    with open(image_path, "rb") as image_file:
        file_name = os.path.basename(image_path)
        files = {'file': (file_name, image_file, 'image/jpeg')} 
        response = requests.post(endpoint_url, files=files)
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.reason}")
    print(response.content)

