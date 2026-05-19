from datetime import date, datetime
from pydantic import BaseModel


class CardRecord(BaseModel):
    id: str
    name: str
    set_id: str
    collector_no: str
    cmc: float | None
    mana_cost: str | None
    card_type: str | None
    colours: list[str]
    colour_identity: list[str]
    keywords: list[str]
    creature_types: list[str]
    rarity: str | None
    text: str | None
    power: str | None
    toughness: str | None
    price_std: float | None
    price_foil: float | None
    price_etched: float | None

    class Config:
        from_attributes = True


class LocationRecord(BaseModel):
    id: int
    name: str
    type: str
    archived: bool
    created_at: datetime | None

    class Config:
        from_attributes = True


class MovementRecord(BaseModel):
    id: int
    copy_id: int
    from_location: str | None
    to_location: str
    moved_at: datetime | None
    reason: str | None

    class Config:
        from_attributes = True


class OwnedCopyRecord(BaseModel):
    id: int
    card: CardRecord
    foil: bool
    etched: bool
    condition: str
    purchase_date: date | None
    purchase_price: float | None
    purchase_source: str | None
    notes: str | None
    current_location: LocationRecord | None

    class Config:
        from_attributes = True


class PoolSummary(BaseModel):
    total_cards: int
    color_counts: dict[str, int]
    avg_cmc: float
    keyword_counts: dict[str, int]


class PoolAnalysisResponse(BaseModel):
    pool_id: str
    cards: list[dict]
    charts: dict[str, str]   # chart_name → Plotly JSON string
    summary: PoolSummary


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str              # pending | running | complete | error
    progress: int            # 0-100
    message: str = ''
