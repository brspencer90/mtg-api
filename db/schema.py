"""
SQLite schema — created via CREATE TABLE IF NOT EXISTS.
No ORM; use db.connection.get_conn() for raw sqlite3 access.
"""

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS cards (
    id              TEXT PRIMARY KEY,   -- Scryfall UUID
    name            TEXT NOT NULL,
    set_id          TEXT NOT NULL,
    collector_no    TEXT NOT NULL,
    cmc             REAL,
    mana_cost       TEXT,
    card_type       TEXT,               -- 'Creature' | 'Non-Creature' | 'Land'
    colours         TEXT,               -- JSON array e.g. '["W","R"]'
    colour_identity TEXT,               -- JSON array
    keywords        TEXT,               -- JSON array
    creature_types  TEXT,               -- JSON array
    rarity          TEXT,
    text            TEXT,
    power           TEXT,
    toughness       TEXT,
    price_std       REAL,
    price_foil      REAL,
    price_etched    REAL,
    last_updated    TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS locations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    type        TEXT DEFAULT 'storage',  -- deck|storage|pool|trade|wishlist
    created_at  TEXT DEFAULT (datetime('now')),
    archived    INTEGER DEFAULT 0        -- 0=active, 1=archived
);

CREATE TABLE IF NOT EXISTS owned_copies (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    card_id         TEXT NOT NULL REFERENCES cards(id),
    foil            INTEGER DEFAULT 0,
    etched          INTEGER DEFAULT 0,
    condition       TEXT DEFAULT 'NM',   -- NM|LP|MP|HP|DMG
    purchase_date   TEXT,                -- ISO date string
    purchase_price  REAL,
    purchase_source TEXT,
    notes           TEXT,
    acquired_at     TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS card_placements (
    copy_id     INTEGER PRIMARY KEY REFERENCES owned_copies(id),
    location_id INTEGER NOT NULL REFERENCES locations(id),
    placed_at   TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS collections (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT UNIQUE NOT NULL,
    set_id     TEXT,
    type       TEXT,    -- 'pool' | 'deck' | 'sideboard' | 'commander'
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS collection_cards (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    collection_id INTEGER REFERENCES collections(id) ON DELETE CASCADE,
    card_id       INTEGER REFERENCES cards(id),
    quantity      INTEGER DEFAULT 1,
    foil          INTEGER DEFAULT 0,
    etched        INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS card_movements (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    copy_id          INTEGER NOT NULL REFERENCES owned_copies(id),
    from_location_id INTEGER REFERENCES locations(id),  -- NULL = newly acquired
    to_location_id   INTEGER NOT NULL REFERENCES locations(id),
    moved_at         TEXT DEFAULT (datetime('now')),
    reason           TEXT
);
"""
