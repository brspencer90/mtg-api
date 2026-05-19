from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from api.services.pool_service import analyze_pool_from_text, analyze_pool_from_ids
from api.models.responses import PoolAnalysisResponse

router = APIRouter(prefix='/pools', tags=['pools'])


@router.post('/analyze', response_model=PoolAnalysisResponse)
async def analyze_pool(
    file: UploadFile = File(...),
    set_id: str | None = Form(default=None),
):
    """
    Upload an MTGA export .txt file (or a plain collector-number .txt file with set_id form field).
    Returns full pool analysis with charts as Plotly JSON.
    """
    raw = (await file.read()).decode('utf-8', errors='replace')

    # Detect format: MTGA export starts with 'Deck'
    if raw.strip().startswith('Deck'):
        try:
            return analyze_pool_from_text(raw, set_id)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
    elif set_id:
        id_list = [line.strip() for line in raw.splitlines() if line.strip()]
        try:
            return analyze_pool_from_ids(id_list, set_id)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
    else:
        raise HTTPException(
            status_code=422,
            detail='For collector-number files, provide set_id as a form field.',
        )
