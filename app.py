# app.py
import os
import tempfile
import time
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

from run_model import Gemma3Generator, CONFIG
print("Starting Gemma3 FastAPI server...")
app = FastAPI(title="Gemma3 Inference Server")
LOG_FILE = "generation_log.txt"

@app.on_event("startup")
def _startup():
    # Load once per process during startup
    print("Starting up Gemma3Generator...")
    with open(LOG_FILE, "a", encoding="utf-8") as log_f:
        log_f.write(f"Starting up Gemma3Generator at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    app.state.generator = Gemma3Generator(CONFIG)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ready")
def ready():
    print("Readiness check...")
    with open(LOG_FILE, "a", encoding="utf-8") as log_f:
        log_f.write(f"Readiness check at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    return {"ready": hasattr(app.state, "generator")}

@app.post("/generate")
async def generate(
    prompt: str = Form(...),
    max_new_tokens: Optional[int] = Form(None),
    files: Optional[List[UploadFile]] = File(None),
):
    if not hasattr(app.state, "generator"):
        return JSONResponse({"error": "Model not ready"}, status_code=503)

    generator: Gemma3Generator = app.state.generator

    temp_paths: List[str] = []
    print("Received generate request with prompt at", time.strftime("%Y-%m-%d %H:%M:%S"))

    with open(LOG_FILE, "a", encoding="utf-8") as log_f:
        log_f.write(f"Received generate request at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_f.write(f"Prompt: {prompt}\n")

    try:
        if files:
            for f in files:
                suffix = os.path.splitext(f.filename)[1] or ".png"
                fd, temp_path = tempfile.mkstemp(suffix=suffix)
                with os.fdopen(fd, "wb") as out:
                    out.write(await f.read())
                temp_paths.append(temp_path)

        output_text = generator.generate(
            prompt=prompt,
            image_paths=temp_paths if temp_paths else None,
            max_new_tokens=max_new_tokens,
        )

        with open(LOG_FILE, "a", encoding="utf-8") as log_f:
            log_f.write(f"Generated output at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_f.write(f"Output: {output_text}\n")

        return JSONResponse({"output": output_text})

    finally:
        for p in temp_paths:
            try:
                os.remove(p)
            except OSError:
                pass
