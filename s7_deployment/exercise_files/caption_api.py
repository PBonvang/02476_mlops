from __future__ import annotations



from http import HTTPStatus

from fastapi import FastAPI, File, UploadFile
from contextlib import asynccontextmanager
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and clean up model on startup and shutdown."""
    global model, feature_extractor, tokenizer, device, gen_kwargs
    print("Loading model")
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    gen_kwargs = {"max_length": 16, "num_beams": 8, "num_return_sequences": 1}

    yield

    print("Cleaning up")
    del model, feature_extractor, tokenizer, device, gen_kwargs


app = FastAPI(lifespan=lifespan)




@app.get("/")
def read_root():
    """Simple root endpoint."""
    return {"Hello": "World"}

@app.post("/caption/")
async def caption_model(data: UploadFile = File(...)):
    """Simple function using vit image captioning."""
    image = Image.open(data.file)
    if image.mode != "RGB":
        image = image.convert(mode="RGB")

    pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    output_ids = model.generate(pixel_values, **gen_kwargs)
    predictions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    predictions = [pred.strip() for pred in predictions]

    return {
        "input": data,
        "output": predictions,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }