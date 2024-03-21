import asyncio
from utils import pipeline, load_model
from fastapi import FastAPI, UploadFile, File

app = FastAPI()

# Load Model
mymodel = load_model("mymodel.pt")

@app.get('/')
def hello():
    return 'Welcome'

@app.post('/innvocations')
async def invoke(upload_file: UploadFile = File(...)):
    contents = await upload_file.read()
    
    output = pipeline(contents, mymodel)
    return output

