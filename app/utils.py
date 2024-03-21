import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, StreamingResponse

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

    # Load image
    image = get_image(contents)

    # Process the image
    processed_image = process(model, image)
    img_byte_array = BytesIO()
    processed_image.save(img_byte_array, format='JPEG')
    img_byte_array.seek(0)

    return StreamingResponse(BytesIO(img_byte_array.read()), media_type="image/jpeg")


    
