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
    with open(os.path.join(BASE_DIR, "index.html"), "r") as f:
        return f.read()


@app.post("/chat")
async def chat(
    message: str = Form(...),
    file: UploadFile = File(...)
):
    upload_path = None

    try:
        print("\n=== NEW REQUEST ===")
        print("Uploaded filename:", file.filename)
        print("User message:", message)

        # 🔹 Save uploaded file
        upload_path = os.path.join(BASE_DIR, f"temp_{uuid.uuid4()}.png")

        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print("Saved image to:", upload_path)

        # 🔹 Load image
        image = cv2.imread(upload_path)

        if image is None:
            raise ValueError("Failed to read image")

        print("Image shape:", image.shape)

        # 🔹 Build graph state
        state = {
            "image": image,
            "mask": None,
            "palette": None,
            "expand": None,
            "result": None,
            "steps": None,
            "message": message
        }

        # 🔥 Run LangGraph
        result_state = graph.invoke(state)

        print("Graph returned keys:", list(result_state.keys()))

        # 🔁 Determine output
        if result_state.get("result") is not None:
            print("Using COLOR result")
            result_image = result_state["result"]

        elif result_state.get("mask") is not None:
            print("Using MASK result")

            mask = result_state["mask"]

            # ✅ Proper binary visualization
            result_image = (mask > 0).astype("uint8") * 255
            result_image = cv2.cvtColor(result_image, cv2.COLOR_GRAY2BGR)

        else:
            raise ValueError("No output generated")

        # ==================================================
        # 🆕 SAVE BEFORE IMAGE
        # ==================================================
        before_filename = f"before_{uuid.uuid4()}.png"
        before_path = os.path.join(OUTPUT_DIR, before_filename)

        cv2.imwrite(before_path, image)

        # ==================================================
        # 🆕 SAVE AFTER IMAGE
        # ==================================================
        after_filename = f"after_{uuid.uuid4()}.png"
        after_path = os.path.join(OUTPUT_DIR, after_filename)

        success = cv2.imwrite(after_path, result_image)
        if not success:
            raise IOError("Failed to save output image")

        print("Saved BEFORE:", before_path)
        print("Saved AFTER:", after_path)

        # 🔁 Cache-busting URLs
        before_url = f"/outputs/{before_filename}?t={uuid.uuid4()}"
        after_url = f"/outputs/{after_filename}?t={uuid.uuid4()}"

        return JSONResponse(
            content={
                "before": before_url,
                "after": after_url
            }
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
        # 🧹 Cleanup temp file
        try:
            if upload_path and os.path.exists(upload_path):
                os.remove(upload_path)
        except Exception as cleanup_error:
            print("Cleanup warning:", cleanup_error)
#to run
# uvicorn main:app --reload
#ctrl C - stop