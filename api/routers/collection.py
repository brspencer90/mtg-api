from fastapi import APIRouter, HTTPException, UploadFile, File
from api.models.requests import (
    AddCopyRequest, MoveCardRequest, CreateLocationRequest, UpdateLocationRequest, UpdateCopyRequest
)
from api.services import collection_service

router = APIRouter(prefix='/collection', tags=['collection'])


@router.get('/copies')
def list_copies(location_id: int | None = None):
    return collection_service.list_copies(location_id)


@router.post('/copies', status_code=201)
def add_copy(req: AddCopyRequest):
    return collection_service.add_copy(
        card_id=req.card_id,
        location_id=req.location_id,
        foil=req.foil,
        etched=req.etched,
        condition=req.condition,
        purchase_date=req.purchase_date,
        purchase_price=req.purchase_price,
        purchase_source=req.purchase_source,
        notes=req.notes,
    )


@router.patch('/copies/{copy_id}')
def update_copy(copy_id: int, req: UpdateCopyRequest):
    result = collection_service.update_copy(copy_id, req.condition, req.notes, req.purchase_price)
    if result is None:
        raise HTTPException(status_code=404, detail='Copy not found')
    return result


@router.delete('/copies/{copy_id}', status_code=204)
def delete_copy(copy_id: int):
    if not collection_service.delete_copy(copy_id):
        raise HTTPException(status_code=404, detail='Copy not found')


@router.get('/copies/{copy_id}/history')
def copy_history(copy_id: int):
    return collection_service.get_copy_history(copy_id)


@router.get('/locations')
def list_locations():
    return collection_service.list_locations()


@router.post('/locations', status_code=201)
def create_location(req: CreateLocationRequest):
    return collection_service.create_location(req.name, req.type)


@router.patch('/locations/{location_id}')
def update_location(location_id: int, req: UpdateLocationRequest):
    result = collection_service.update_location(location_id, req.name, req.archived)
    if result is None:
        raise HTTPException(status_code=404, detail='Location not found')
    return result


@router.post('/move')
def move_card(req: MoveCardRequest):
    result = collection_service.move_card(req.copy_id, req.to_location_id, req.reason)
    if result is None:
        raise HTTPException(status_code=404, detail='Copy not found')
    return result


@router.post('/import', status_code=202)
async def import_file(file: UploadFile = File(...)):
    """Upload a .txt collector-number file to import into the collection."""
    import os, tempfile
    from pathlib import Path
    from db.connection import get_conn
    from db.migrations.import_legacy import import_file as do_import

    content = await file.read()
    stem = Path(file.filename).stem
    suffix = Path(file.filename).suffix

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, prefix=stem + '_') as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        conn = get_conn()
        count = do_import(conn, tmp_path)
        conn.close()
    finally:
        os.unlink(tmp_path)

    return {'imported': count, 'filename': file.filename}
