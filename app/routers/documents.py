from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from app.config import get_settings
from app.documents.extract import SUPPORTED_EXTENSIONS
from app.documents.service import upload_document
from app.rate_limit import limiter

router = APIRouter()

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB -- well under Anthropic's own 32MB request limit


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int


@router.post("/documents")
@limiter.limit(get_settings().rate_limit_documents)
async def upload(request: Request, file: UploadFile = File(...)) -> UploadResponse:  # noqa: B008 -- FastAPI's own idiom
    filename = file.filename or "unnamed"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(status_code=400, detail=f"不支援的檔案格式:{ext or filename}(支援:{supported})")

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="檔案過大(上限 20MB)")

    try:
        result = await upload_document(filename, content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return UploadResponse(**result)
