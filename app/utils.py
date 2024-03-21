import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
from fastapi.responses import JSONResponse, StreamingResponse

def get_image(contents):
    try:
        image = np.array(Image.open(BytesIO(contents)))
        return image
    except Exception as e:
        return JSONResponse(content=f"Error processing image: {e}", status_code=500)

def load_model(file_path):
    try:
        model = torch.jit.load(file_path)
        return model
    except Exception as e:
        return JSONResponse(content=f"Error loading model: {e}", status_code=500)

def process(model, image):
    try:
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
    except Exception as e:
        return JSONResponse(content=f"Error processing image: {e}", status_code=500)

def pipeline(contents, model):
    try:
        # Load image
        image = get_image(contents)
        if isinstance(image, JSONResponse):
            return image  # Return error response if get_image failed

        # Process the image
        processed_image = process(model, image)
        img_byte_array = BytesIO()
        processed_image.save(img_byte_array, format='JPEG')
        img_byte_array.seek(0)

        return StreamingResponse(BytesIO(img_byte_array.read()), media_type="image/jpeg")
    except Exception as e:
        return JSONResponse(content=f"Error in pipeline: {e}", status_code=500)
