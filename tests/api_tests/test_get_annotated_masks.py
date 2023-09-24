import requests
import os
import json
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def img_grey(data):
    return Image.fromarray(data * 255, mode='L').convert('1')


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = np.array(ann['segmentation'])
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))


def plot_results(outputs, image_ori, save_path='../vis/'):
    """
    plot input image and its reuslts
    """
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    fig = plt.figure()
    plt.imshow(image_ori)
    plt.savefig('input.png')
    show_anns(outputs)
    fig.canvas.draw()
    im = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    plt.savefig('example.png')
    return im


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


    # with open("test.json", "r") as json_file:
    #     response = json.load(json_file)

    # base_image = Image.open(image_path)
    # for food in response:
    #     newim = base_image.crop(food['box'])
    #     print(newim.size)
    #     img_grey(np.array(food['masks'][1]['segmentation'])).save('test.png')
    #     # plot_results(food['masks'], newim, save_path='./')


    
    # print(response)

    # # Save the response content to a JSON file
    response_data = response.json()
    output_path = "test.json"  # Replace with your desired path and file name
    with open(output_path, "w") as json_file:
        json.dump(response_data, json_file, indent=4)
    print(f"Saved response to {output_path}")
