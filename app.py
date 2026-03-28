from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import shutil

from agent import run_agent
from tools.segmentation import run_segmentation

app = FastAPI()

UPLOAD_PATH = "temp_input.png"
OUTPUT_PATH = "outputs/result.png"


@app.post("/chat")
async def chat(
    message: str = Form(...),
    file: UploadFile = File(...)
):
    # Save uploaded image
    with open(UPLOAD_PATH, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Decide action
    action = run_agent(message)

    if action["tool"] == "segment":
        result_path = run_segmentation(UPLOAD_PATH, OUTPUT_PATH)
        return FileResponse(result_path, media_type="image/png")

    return {"message": "No action taken"}