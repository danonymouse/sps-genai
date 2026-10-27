from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from PIL import UnidentifiedImageError
from pydantic import BaseModel, Field

from app.bigram_model import BigramModel
from app.digit_generator import DigitGenerator
from app.embedding_model import EmbeddingModel
from app.image_classifier import CifarImagePredictor

app = FastAPI()

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]

# Load each model once at startup
bigram_model = BigramModel(corpus)
embedding_model = EmbeddingModel()          # Assignment 1
image_predictor = CifarImagePredictor()     # Assignment 2
digit_generator = DigitGenerator()          # Assignment 3

# The image formats /classify-image accepts
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png"}


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


class EmbeddingRequest(BaseModel):
    word: str = Field(min_length=1, examples=["apple"])


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    embedding = embedding_model.calculate_embedding(request.word)
    if embedding is None:
        raise HTTPException(
            status_code=404,
            detail=f"No embedding found for '{request.word}' (not in vocabulary).",
        )
    return {
        "word": request.word,
        "dimension": len(embedding),
        "embedding": embedding,
    }


# Image classification (Assignment 2)
@app.post(
    "/classify-image",
    responses={
        400: {"description": "Empty or unreadable image file"},
        415: {"description": "File is not a JPEG or PNG"},
    },
)
async def classify_image(
    file: UploadFile = File(..., description="A JPEG or PNG image"),
    top_k: int = Query(3, ge=1, le=10, description="How many ranked classes to return"),
):
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Upload a JPEG or PNG.",
        )
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    try:
        ranked = image_predictor.top_predictions(image_bytes, k=top_k)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not decode the file as an image.")
    return {
        "filename": file.filename,
        "prediction": ranked[0]["label"],
        "top_predictions": ranked,
    }


# Handwritten digit generation with the GAN (Assignment 3)
@app.get(
    "/generate-digits",
    response_class=Response,
    responses={200: {"content": {"image/png": {}},
                     "description": "A PNG grid of generated handwritten digits"}},
)
def generate_digits(
    num_images: int = Query(16, ge=1, le=64, description="How many digits to generate"),
    seed: int | None = Query(None, description="Optional: the same seed gives the same digits"),
):
    png = digit_generator.generate_png(num_images=num_images, seed=seed)
    return Response(content=png, media_type="image/png")