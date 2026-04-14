from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
import shutil
import os
import uuid

from agent import run_agent

app = FastAPI()

# Ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)


@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html") as f:
        return f.read()


@app.post("/chat")
async def chat(
    message: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        # 🔹 Create unique filename (avoid overwrite bugs)
        upload_path = f"temp_{uuid.uuid4()}.png"

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

        if not os.path.exists(result_path):
            return JSONResponse(
                content={
                    "error": "Output file does not exist",
                    "result": result_path
                },
                status_code=500
            )

        # ✅ Return image
        return FileResponse(result_path, media_type="image/png")

    except Exception as e:
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