import io
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from backend.app.pipeline import VerificationPipeline
from backend.app.schemas import VerificationResponse

logging.basicConfig(level=logging.INFO)

pipeline: VerificationPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    pipeline = VerificationPipeline()
    yield


app = FastAPI(title="AI Identity Shield API", lifespan=lifespan)

# Local dev frontend (Vite default). Adjust/add origins as needed - no
# secrets here, just CORS allowlisting.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/verify-document", response_model=VerificationResponse)
async def verify_document(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="uploaded file is not a readable image")

    result = pipeline.verify(image)
    return {"filename": file.filename, **result}
