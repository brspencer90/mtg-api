from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.core.config import settings
from api.routers import collection, pools, sets, analysis, import_router
from api.services.collection_service import ensure_default_locations
from db.connection import init_db

app = FastAPI(title='MTG Analysis API', version='0.1.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(collection.router)
app.include_router(pools.router)
app.include_router(sets.router)
app.include_router(analysis.router)
app.include_router(import_router.router)


@app.on_event('startup')
def on_startup():
    init_db()
    ensure_default_locations()


@app.get('/health')
def health():
    return {'status': 'ok'}
