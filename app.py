from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os
import uuid

from agent import run_agent

app = FastAPI()

# 📁 Base directory (CRITICAL FIX)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STATIC_DIR = os.path.join(BASE_DIR, "static")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# 📁 Ensure folders exist
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 🌐 Serve static files (welcome image)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# 🌐 Serve generated outputs
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
    try:
        # 🔹 Create unique filename
        upload_path = os.path.join(BASE_DIR, f"temp_{uuid.uuid4()}.png")

        # Save uploaded image
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print("\n=== NEW REQUEST ===")
        print("User message:", message)
        print("Saved image to:", upload_path)

        # 🔹 Run agent
        result_path = run_agent(message, upload_path)

        print("Returned path from agent:", result_path)

        # 🔥 Validate output
        if not isinstance(result_path, str):
            return JSONResponse(
                content={
                    "error": "Agent did not return a string path",
                    "result": str(result_path)
                },
                status_code=500
            )

        # Ensure absolute path
        if not os.path.isabs(result_path):
            result_path = os.path.join(BASE_DIR, result_path)

        if not os.path.exists(result_path):
            return JSONResponse(
                content={
                    "error": "Output file does not exist",
                    "result": result_path
                },
                status_code=500
            )

        # Convert to URL path (important for frontend)
        relative_path = os.path.relpath(result_path, OUTPUT_DIR)
        url_path = f"/outputs/{relative_path}"

        # 🧹 Cleanup temp file
        try:
            if os.path.exists(upload_path):
                os.remove(upload_path)
        except Exception as cleanup_error:
            print("Cleanup warning:", cleanup_error)

        # ✅ Return URL path
        return JSONResponse(
            content={
                "result": url_path
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
#to run
# uvicorn app:app --reload
#ctrl C - stop