import os
import json
import tempfile
import logging
import asyncio
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from PIL import Image

from main import PDFParser
from utils.OpenAIProviders import create_llm_provider, create_vlm_provider
from config_manager import load_settings, save_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DocuSenseAPI")

app = FastAPI(
    title="DocuSense API",
    description="Multi-Model AI PDF Content Extraction & Analysis Server",
    version="0.1.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class PDFUrlRequest(BaseModel):
    pdf_url: str = Field(..., description="HTTP/HTTPS URL or Base64 data URI of the PDF document")
    process_sequentially: Optional[bool] = None
    ocr_fallback: Optional[bool] = None
    page_selection: Optional[str] = Field(None, description="Page selection range (e.g. '1-5, 8, 10-12')")


class SettingsModel(BaseModel):
    llm_provider: str = "gemini"
    llm_api_key: Optional[str] = ""
    llm_base_url: Optional[str] = ""
    llm_model: Optional[str] = ""
    vlm_provider: str = "gemini"
    vlm_api_key: Optional[str] = ""
    vlm_base_url: Optional[str] = ""
    vlm_model: Optional[str] = ""
    process_sequentially: bool = True
    ocr_fallback: bool = True
    max_workers: int = 4


class TestConnectionRequest(BaseModel):
    type: str  # "llm" or "vlm"
    provider: str
    api_key: Optional[str] = ""
    base_url: Optional[str] = ""
    model: Optional[str] = ""


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the single page frontend web application."""
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>DocuSense API Server is running</h1><p>Visit <a href='/docs'>/docs</a> for API documentation.</p>")


@app.get("/api/health")
async def health_check():
    """Health check endpoint returning configuration status."""
    cfg = load_settings()
    return {
        "status": "online",
        "app": "DocuSense",
        "llm_provider": cfg.get("llm_provider"),
        "llm_model": cfg.get("llm_model"),
        "vlm_provider": cfg.get("vlm_provider"),
        "vlm_model": cfg.get("vlm_model")
    }


@app.get("/api/settings")
async def get_settings():
    """Get currently saved settings."""
    cfg = load_settings()
    return JSONResponse(content=cfg)


@app.post("/api/settings")
async def update_settings(settings: SettingsModel):
    """Update settings and save to settings.json."""
    try:
        updated = save_settings(settings.model_dump())
        return JSONResponse(content={"status": "success", "settings": updated})
    except Exception as e:
        logger.error(f"Error saving settings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save settings: {str(e)}")


@app.post("/api/settings/test")
async def test_connection(req: TestConnectionRequest):
    """Test LLM or VLM provider connection with given API key and model."""
    try:
        if req.type.lower() == "llm":
            llm = create_llm_provider(
                provider_type=req.provider,
                api_key=req.api_key,
                base_url=req.base_url,
                model_name=req.model
            )
            text, summary = llm.process_text("Ping test for connection verification.")
            if "Error" in summary or "Error" in text:
                return JSONResponse(content={"status": "error", "message": f"{summary} | {text}"})
            return JSONResponse(content={"status": "success", "message": f"Connected! Response summary: '{summary[:100]}...'"})

        elif req.type.lower() == "vlm":
            vlm = create_vlm_provider(
                provider_type=req.provider,
                api_key=req.api_key,
                base_url=req.base_url,
                model_name=req.model
            )
            test_img = Image.new("RGB", (100, 100), color="white")
            text, summary = vlm.process_image(test_img)
            if "Error" in summary or "Error" in text:
                return JSONResponse(content={"status": "error", "message": f"{summary} | {text}"})
            return JSONResponse(content={"status": "success", "message": f"Connected! Response summary: '{summary[:100]}...'"})

        else:
            raise HTTPException(status_code=400, detail="Invalid test type. Must be 'llm' or 'vlm'")
    except Exception as e:
        logger.error(f"Connection test failed: {e}", exc_info=True)
        return JSONResponse(content={"status": "error", "message": str(e)})


@app.post("/api/parse-pdf")
async def parse_pdf(
    file: Optional[UploadFile] = File(None),
    pdf_url: Optional[str] = Form(None),
    process_sequentially: Optional[bool] = Form(None),
    ocr_fallback: Optional[bool] = Form(None),
    page_selection: Optional[str] = Form(None),
    body: Optional[PDFUrlRequest] = Body(None)
):
    """
    Parse a PDF from a File upload, URL string, or Base64 string.
    Supports both Multipart Form Data and JSON body payloads.
    """
    cfg = load_settings()

    # Determine input URL or file path
    target_input = None
    temp_filepath = None

    if file:
        try:
            suffix = ".pdf"
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            contents = await file.read()
            temp_file.write(contents)
            temp_file.close()
            temp_filepath = temp_file.name
            target_input = temp_filepath
            logger.info(f"Received uploaded file '{file.filename}', saved to '{temp_filepath}'")
        except Exception as e:
            logger.error(f"Error handling uploaded file: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {str(e)}")

    elif pdf_url:
        target_input = pdf_url
    elif body and body.pdf_url:
        target_input = body.pdf_url

    if not target_input:
        raise HTTPException(status_code=400, detail="PDF input required (file upload, pdf_url form param, or JSON body).")

    # Determine runtime options
    seq_proc = process_sequentially if process_sequentially is not None else (body.process_sequentially if body and body.process_sequentially is not None else cfg.get("process_sequentially", True))
    ocr_fb = ocr_fallback if ocr_fallback is not None else (body.ocr_fallback if body and body.ocr_fallback is not None else cfg.get("ocr_fallback", True))
    page_sel = page_selection or (body.page_selection if body else None)

    try:
        # Initialize providers dynamically based on current configuration
        llm_provider = create_llm_provider(
            provider_type=cfg.get("llm_provider", "gemini"),
            api_key=cfg.get("llm_api_key"),
            base_url=cfg.get("llm_base_url"),
            model_name=cfg.get("llm_model")
        )

        vlm_provider = create_vlm_provider(
            provider_type=cfg.get("vlm_provider", "gemini"),
            api_key=cfg.get("vlm_api_key"),
            base_url=cfg.get("vlm_base_url"),
            model_name=cfg.get("vlm_model")
        )

        parser = PDFParser(
            llm_provider=llm_provider,
            vlm_provider=vlm_provider,
            process_sequentially=seq_proc,
            max_workers=cfg.get("max_workers", 4),
            min_image_size=cfg.get("min_image_size", 100),
            ocr_fallback=ocr_fb
        )

        result = parser.parse_pdf(target_input, page_selection=page_sel)

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return {"status": "success", "data": result}

    except Exception as e:
        logger.error(f"Error during PDF parsing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"PDF parsing error: {str(e)}")

    finally:
        # Clean up temp upload file if created
        if temp_filepath and os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
            except Exception as e:
                logger.warning(f"Could not remove temp file '{temp_filepath}': {e}")


# Legacy route support for backward compatibility with previous api.py
@app.post("/parse-pdf/")
async def parse_pdf_legacy(request: PDFUrlRequest):
    return await parse_pdf(body=request)


@app.post("/api/parse-pdf-stream")
async def parse_pdf_stream(
    file: Optional[UploadFile] = File(None),
    pdf_url: Optional[str] = Form(None),
    process_sequentially: Optional[bool] = Form(None),
    ocr_fallback: Optional[bool] = Form(None),
    page_selection: Optional[str] = Form(None)
):
    """
    Streaming endpoint providing real-time Server-Sent Events (SSE) progress logs
    and result delivery during PDF parsing.
    """
    cfg = load_settings()

    target_input = None
    temp_filepath = None

    if file:
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            contents = await file.read()
            temp_file.write(contents)
            temp_file.close()
            temp_filepath = temp_file.name
            target_input = temp_filepath
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {str(e)}")
    elif pdf_url:
        target_input = pdf_url

    if not target_input:
        raise HTTPException(status_code=400, detail="PDF input required (file upload or pdf_url form param).")

    seq_proc = process_sequentially if process_sequentially is not None else cfg.get("process_sequentially", True)
    ocr_fb = ocr_fallback if ocr_fallback is not None else cfg.get("ocr_fallback", True)

    async def event_generator():
        loop = asyncio.get_event_loop()
        queue = asyncio.Queue()

        def status_cb(msg, level="info"):
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "log", "message": msg, "level": level})

        def run_parse():
            try:
                llm = create_llm_provider(
                    provider_type=cfg.get("llm_provider", "gemini"),
                    api_key=cfg.get("llm_api_key"),
                    base_url=cfg.get("llm_base_url"),
                    model_name=cfg.get("llm_model")
                )
                vlm = create_vlm_provider(
                    provider_type=cfg.get("vlm_provider", "gemini"),
                    api_key=cfg.get("vlm_api_key"),
                    base_url=cfg.get("vlm_base_url"),
                    model_name=cfg.get("vlm_model")
                )

                parser = PDFParser(
                    llm_provider=llm,
                    vlm_provider=vlm,
                    process_sequentially=seq_proc,
                    max_workers=cfg.get("max_workers", 4),
                    min_image_size=cfg.get("min_image_size", 100),
                    ocr_fallback=ocr_fb
                )

                res = parser.parse_pdf(target_input, page_selection=page_selection, status_callback=status_cb)
                if "error" in res:
                    loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "message": res["error"]})
                else:
                    loop.call_soon_threadsafe(queue.put_nowait, {"type": "result", "data": res})
            except Exception as e:
                logger.error(f"Error in streaming parse: {e}", exc_info=True)
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "message": str(e)})

        loop.run_in_executor(None, run_parse)

        while True:
            item = await queue.get()
            yield f"data: {json.dumps(item)}\n\n"
            if item["type"] in ("result", "error"):
                break

        # Cleanup temp file
        if temp_filepath and os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
            except Exception:
                pass

    return StreamingResponse(event_generator(), media_type="text/event-stream")

