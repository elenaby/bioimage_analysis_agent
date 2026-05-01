from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os
import uuid
import cv2
import numpy as np

from graph import build_graph

app = FastAPI()

# 🔧 Initialize graph once
graph = build_graph()

# 📁 Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STATIC_DIR = os.path.join(BASE_DIR, "static")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


@app.get("/", response_class=HTMLResponse)
def home():
    index_path = os.path.join(BASE_DIR, "index.html")
    with open(index_path, "r") as f:
        return f.read()


@app.post("/chat")
async def chat(
    message: str = Form(...),
    file: UploadFile = File(...)
):
    upload_path = None

    try:
        # 🔹 Save uploaded file
        upload_path = os.path.join(BASE_DIR, f"temp_{uuid.uuid4()}.png")

        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print("\n=== NEW REQUEST ===")
        print("User message:", message)
        print("Saved image to:", upload_path)

        # 🔹 Load image
        image = cv2.imread(upload_path)

        if image is None:
            raise ValueError("Failed to read image")

        # 🔹 Build graph state (UPDATED)
        state = {
            "image": image,
            "mask": None,
            "palette": None,
            "expand": None,
            "result": None,
            "steps": None,        # 👈 required for dynamic routing
            "message": message    # 👈 required for LLM
        }

        # 🔥 Run LangGraph
        result_state = graph.invoke(state)

        # 🔁 Handle cases where color step was skipped
        if result_state.get("result") is not None:
            result_image = result_state["result"]
        elif result_state.get("mask") is not None:
            # fallback: return mask if no color step
            mask = result_state["mask"]
            result_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        else:
            raise ValueError("No output generated")

        # 🔹 Save output image
        output_filename = f"result_{uuid.uuid4()}.png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        success = cv2.imwrite(output_path, result_image)
        if not success:
            raise IOError("Failed to save output image")

        print("Saved output to:", output_path)

        # 🔁 Convert to URL
        url_path = f"/outputs/{output_filename}"

        return JSONResponse(
            content={"result": url_path}
        )

    except Exception as e:
        print("❌ SERVER ERROR:", str(e))
        return JSONResponse(
            content={
                "error": "Server error",
                "details": str(e)
            },
            status_code=500
        )

    finally:
        # 🧹 Always cleanup temp file
        try:
            if upload_path and os.path.exists(upload_path):
                os.remove(upload_path)
        except Exception as cleanup_error:
            print("Cleanup warning:", cleanup_error)
#to run
# uvicorn main:app --reload
#ctrl C - stop