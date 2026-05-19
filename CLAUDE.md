# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Full-stack MTG collection manager and card analysis tool. The project has two layers:

1. **Python analysis toolkit** (original) — fetches card data from Scryfall, encodes attributes into a pandas DataFrame, and generates Plotly visualizations for sealed/draft pool evaluation.
2. **Web application** (added) — FastAPI backend + Vite/React frontend for managing a physical card collection.

### Current state (as of 2026-05-19)

The web app is the active focus. Key features built so far:

- **Collection browser** — paginated table of owned cards, sortable by name/set/type/rarity/price/condition/location, filterable by set, location, rarity, card type, color. Inline edit modal (updates condition, foil, etched, purchase info; refreshes price from Scryfall). Move modal to relocate cards between locations.
- **Card adder** — three modes: by name (Scryfall autocomplete → printing picker), by set+ID (collector number lookup), and bulk add (paste a list of collector numbers, preview, then save). All modes support inline location creation via `LocationSelect`.
- **Location management** — physical locations tracked in DB. Default locations seeded on startup: "Card Pool" (pool), "Storage" (storage), "Trade Pile" (trade). Old import-source location names (e.g. "BLB Play Booster") are stored as `purchase_source` on each copy after the migration.
- **Moxfield deck import** — SSE-streaming import that fetches each card from Scryfall and adds it to the collection. Preview endpoint returns deck metadata without writing to DB.
- **Movement history** — every card placement change is logged in `card_movements`.

### Data migration note

All 1,669 existing cards were migrated to "Black Storage Box" (physical storage location). The old location names (e.g. "BLB Play Booster", "DSK Sealed Pool") now live in `owned_copies.purchase_source` as provenance/source info. Old source-style locations are archived in the DB (won't appear in UI) but remain for movement history integrity.

## Environment Setup

The virtual environment lives at `.env_mtg/`. Activate it before running any scripts:

```powershell
.env_mtg\Scripts\Activate.ps1
```

Install dependencies if needed:

```powershell
pip install pandas numpy plotly requests scikit-learn nltk sqlalchemy fastapi uvicorn
```

NLTK stopwords must also be downloaded once:

```python
import nltk; nltk.download('stopwords'); nltk.download('punkt')
```

### Running the web app

**Backend** (FastAPI, port 8000):
```powershell
uvicorn api.main:app --reload
```

**Frontend** (Vite + React, port 5173):
```powershell
cd frontend && npm run dev
```

The frontend proxies `/api`, `/collection`, and `/pools` to `localhost:8000` via `vite.config.ts`.

**Database** — SQLite at `db/mtg.db` (gitignored; copy manually between machines). Initialized automatically on first backend startup. To re-run the legacy card data import:
```powershell
python -m db.migrations.import_legacy
```

## Running the Code

There is no entry-point script. The files use Jupyter-style `# %%` cell markers and are intended to be run interactively (e.g., in VS Code with the Jupyter extension, or directly in JupyterLab). Execute cells from `mtg.py` as the primary workflow driver.

Typical session flow:

```python
# Fetch and encode an entire set (uses Scryfall API, ~0.1s per card)
df = get_all_from_set('dsk')

# Or parse a local deck file
df = pull_parse_file('dsk', source='deck')     # reads Deck.txt
df = pull_parse_file('dsk', source='mtga')     # reads mtga_export.txt
df = pull_parse_file('dsk', source='card data/my_pool.csv')

# Visualize
visualize_deck(df, 'dsk')
```

## Architecture

### Data Pipeline

```
Scryfall API → get_card_info() → encode_features() → visualize_deck() / plot_set_specific()
```

1. **`mtg_func.py`** — Core fetch/encode layer
   - `get_card_info(id, set_id)` hits `api.scryfall.com/cards/{set_id}/{id}` and returns a flat list matching `Constants.col`
   - After fetching, set-specific keywords are injected by calling `otj_keywords()`, `mh3_keywords()`, `dsk_keywords()` from `mtg_keywords.py`
   - `encode_features(df)` one-hot encodes Colors, Keywords, and Creature Types using `MultiLabelBinarizer`; also derives `Card Type`, `Colour`, and per-color mana pip counts

2. **`constants.py`** — Single source of truth for column names (`Constants.col`), color mappings, keyword lists, archetype definitions, and letter-grade score mappings. `Constants.expansion_list` is built at import time by reading `mtga_card_data.csv`.

3. **`mtg_keywords.py`** — Set-specific mechanic detection via regex. Each function (`otj_keywords`, `mh3_keywords`, `dsk_keywords`) appends custom keyword strings (e.g., `'Crime'`, `'Energy'`, `'Eerie'`) to the card's keyword list. Handles double-faced cards separately via the `double_face` flag.

4. **`mtg_plot.py`** — Visualization layer. `plot_set_specific(df, set_id)` dispatches to set-specific plot functions (`otj_plots`, `mh3_plots`, `blb_plots`, `dsk_plots`). All charts use Plotly with `template='plotly_dark'`.

5. **`mtg.py`** — Orchestration layer and notebook entry point. Contains `get_all_from_set`, `pull_parse_file`, `visualize_deck`, and `get_word_frequency` (NLTK-based card text analysis).

6. **`17lands.py`** — Downloads game/draft data from the `17lands-public` S3 bucket. Separate from the main pipeline; used for meta-analysis.

7. **`mtga-mtg_card.py`** — Connects to the local MTGA client's SQLite database via SQLAlchemy for local card lookup.

### Adding Support for a New MTG Set

1. Add a new keyword detection function in `mtg_keywords.py` following the pattern of `dsk_keywords()`
2. Call the new function inside `get_card_info()` in `mtg_func.py` (after existing keyword calls)
3. Add archetype keys/creature lists to `Constants` in `constants.py` if the set has unique archetypes
4. Add a plot function in `mtg_plot.py` and register it in the `plot_set_specific()` dispatcher

### Input File Formats

- **`Deck.txt`** — One Scryfall collector number per line (plain integers; append `f` for foil, `e` for etched)
- **`mtga_export.txt`** — Standard MTGA export format: `1 Card Name (SET) 123`
- **`card data/*.csv`** — Sealed pool CSVs; typically one collector number per row, no header

### Key API

- Scryfall: `https://api.scryfall.com` — rate-limited via `time.sleep(0.1)` per card in `get_card_info`
- 17Lands S3: `https://17lands-public.s3.amazonaws.com` — bulk download in `17lands.py`

## Web App Architecture

### Backend (`api/`)

```
api/
  main.py                    # FastAPI app, CORS, router registration, startup hooks
  routers/
    collection.py            # CRUD for copies, locations, movements
    import_router.py         # Moxfield preview + SSE-streaming save
    pools.py                 # Sealed/draft pool analysis
    sets.py                  # Set-level analysis
    analysis.py              # Chart endpoints
  services/
    collection_service.py    # All DB reads/writes (plain sqlite3, no ORM)
  models/
    requests.py              # Pydantic input models
  core/
    config.py                # CORS settings
```

**DB schema** (5 tables, `db/mtg.db`):
- `cards` — Scryfall card catalog (UUID PK, name, set_id, collector_no, prices, etc.)
- `owned_copies` — one row per physical card you own (foil, condition, purchase info)
- `locations` — named physical places (type: pool/deck/storage/trade/wishlist; archived flag)
- `card_placements` — current location of each copy (1:1 with owned_copies)
- `card_movements` — audit log of every location change

**Important conventions:**
- Plain `sqlite3` (stdlib) only — no SQLAlchemy ORM
- Sort columns are whitelisted in `_SORT_COLS` dict to prevent SQL injection
- `ensure_default_locations()` is called on startup to seed Card Pool, Storage, Trade Pile
- `purchase_source` on `owned_copies` stores where the card came from (e.g. "BLB Play Booster"); `location_id` in `card_placements` tracks where it physically lives now

### Frontend (`frontend/`)

```
frontend/src/
  App.tsx                    # Tab shell: Collection | Import Deck
  pages/
    Collection.tsx           # Sub-tabs: Browse | Add Card
  components/
    CollectionBrowser.tsx    # Sortable/filterable table, edit + move modals
    CollectionAdder.tsx      # By Name / By Set+ID / Bulk Add modes
    LocationSelect.tsx       # Reusable dropdown with inline "+ New" creation
    ImportDeck.tsx           # Moxfield URL import with SSE progress bar
  api/
    client.ts                # Typed fetch wrappers for all endpoints
```
