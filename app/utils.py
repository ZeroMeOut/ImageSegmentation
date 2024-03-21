import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
import json 

## Has to be a better way to return the errors
## It was just used for testing btw

def get_image(contents):
    image = np.array(Image.open(BytesIO(contents)))
    return image
    

def load_model(file_path):
    model = torch.jit.load(file_path)
    return model 
    

def process(model, image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((1024, 1024)),
        transforms.ToTensor()
    ])

    image = transform(image).unsqueeze(0) 
    
    # Checking for alpha channels (especially png)
    if image.shape[1] != 3:
        image = image[:, :3, :, :]

    # Input the image to the model

    output = model(image)
    imclass = torch.argmax(output, dim=1)[0,:,:]
    imclass_np = imclass.cpu().numpy()
    processed_image = Image.fromarray((imclass_np * 255).astype(np.uint8)) 

    return processed_image

def pipeline(contents, model):
    try: 
        # Load image
        image = get_image(contents)

        # Process the image
        processed_image = process(model, image)

        return type(processed_image)
    
    except Exception as e:
        return f"Error processing image from pipleline: {e}"


    
