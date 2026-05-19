from fastapi import APIRouter, HTTPException, UploadFile, File
from api.models.requests import (
    AddCopyRequest, MoveCardRequest, CreateLocationRequest, UpdateLocationRequest, UpdateCopyRequest
)
from api.services import collection_service

router = APIRouter(prefix='/collection', tags=['collection'])


@router.get('/copies')
def list_copies(
    search: str | None = None,
    set_id: str | None = None,
    location_id: int | None = None,
    rarity: str | None = None,
    card_type: str | None = None,
    color: str | None = None,
    sort_by: str = 'name',
    sort_dir: str = 'asc',
    page: int = 1,
    page_size: int = 50,
):
    return collection_service.list_copies(
        search=search, set_id=set_id, location_id=location_id,
        rarity=rarity, card_type=card_type, color=color,
        sort_by=sort_by, sort_dir=sort_dir,
        page=page, page_size=page_size,
    )


@router.get('/filter-options')
def filter_options():
    """Return distinct sets and locations for filter dropdowns."""
    return {
        'sets': collection_service.list_owned_sets(),
        'locations': collection_service.list_locations(),
    }


@router.get('/ensure-card')
def ensure_card(scryfall_id: str | None = None,
                set_id: str | None = None,
                collector_no: str | None = None):
    """Ensure a card row exists in the DB (fetch from Scryfall if needed). Returns the card."""
    card = collection_service.ensure_card(scryfall_id, set_id, collector_no)
    if card is None:
        raise HTTPException(status_code=404, detail='Card not found on Scryfall')
    return card


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
    result = collection_service.update_copy(
        copy_id, req.condition, req.notes, req.purchase_price,
        req.foil, req.etched, req.purchase_date, req.purchase_source,
    )
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
