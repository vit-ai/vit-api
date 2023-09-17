import requests
import sys
import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
def draw_boxes_on_image(image_path, boxes_and_conf):
    # Load the image using PIL
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    # If you want to use a custom font, specify the font file
    # Otherwise, a default font is used (might be small in some versions of Pillow)
    # font = ImageFont.truetype('path_to_font.ttf', size=15)
    font = ImageFont.load_default()

    # Loop through boxes and confidences and draw them on the image
    boxes, confidences = boxes_and_conf
    for box, conf in zip(boxes, confidences):
        # Draw the bounding box
        if conf > 0.1:
            draw.rectangle(box, outline="red", width=2)
        
        # Draw the confidence value
        label = f"{conf:.2f}"
        label_w, label_h = draw.textsize(label, font=font)
        # Positioning the label at the top-left corner of the box
        label_position = (box[0], box[1] - label_h if box[1] - label_h > 0 else box[1])
        draw.text(label_position, label, fill="red", font=font)

    # Save the image with boxes and confidences drawn
    output_path = "/Users/liammckenna/vit-ai/api/vit-api/tests/gpu_tests/results/yolo_output.jpg"
    img.save(output_path)
    return img





def call_endpoint(image_path):
    # Check if the file exists
    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' does not exist.")
        return

    # Define the endpoint URL
    endpoint_url = "https://liam-mckenna04--yolo-inference-dev.modal.run" 

    # Open the image file in binary mode
    with open(image_path, "rb") as image_file:
        # Prepare the multipart request
        file_name = os.path.basename(image_path)

        files = {'uploadfile': (file_name, image_file, 'image/jpeg')}
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
    content = call_endpoint(img_path)
    draw_boxes_on_image(img_path, content)
