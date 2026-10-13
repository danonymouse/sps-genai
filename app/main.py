from fastapi import FastAPI, File, HTTPException, Query, UploadFile  # NEW: File, Query, UploadFile
from PIL import UnidentifiedImageError  # NEW
from pydantic import BaseModel, Field

from app.bigram_model import BigramModel
from app.embedding_model import EmbeddingModel
from app.image_classifier import CifarImagePredictor  # NEW

app = FastAPI()

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]

bigram_model = BigramModel(corpus)

# Load the spaCy model once at startup (takes a few seconds)
embedding_model = EmbeddingModel()

# NEW: load the trained CNN once at startup
image_predictor = CifarImagePredictor()

# NEW: the image formats /classify-image accepts
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


# NEW: image classification endpoint (Assignment 2)
@app.post("/classify-image")
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