from typing import Literal
from pydantic import BaseModel


class MoveCardRequest(BaseModel):
    copy_id: int
    to_location_id: int
    reason: str = ''


class CreateLocationRequest(BaseModel):
    name: str
    type: Literal['deck', 'storage', 'pool', 'trade', 'wishlist'] = 'storage'


class UpdateLocationRequest(BaseModel):
    name: str | None = None
    archived: bool | None = None


class AddCopyRequest(BaseModel):
    card_id: str           # Scryfall UUID
    location_id: int
    foil: bool = False
    etched: bool = False
    condition: Literal['NM', 'LP', 'MP', 'HP', 'DMG'] = 'NM'
    purchase_date: str | None = None   # ISO date string
    purchase_price: float | None = None
    purchase_source: str | None = None
    notes: str | None = None


class UpdateCopyRequest(BaseModel):
    condition: Literal['NM', 'LP', 'MP', 'HP', 'DMG'] | None = None
    notes: str | None = None
    purchase_price: float | None = None
