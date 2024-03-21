from utils import *
import requests
import os

file_path = "../models/mymodel.pt"
image_url = 'https://www.showbizpizza.com/photos/cec/tx_roundrock/14.jpg'

response = requests.get(image_url)
image_bytes = response.content

# Load image
image = get_image(image_bytes)

# Load Model
model = load_model(file_path)

# Process the image
processed_image = process(model, image)

# Save image
save_dir = 'saved_image'
filename = f'processed_image_{image_url[-5:]}.jpg'

os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, filename)
processed_image.save(save_path)

