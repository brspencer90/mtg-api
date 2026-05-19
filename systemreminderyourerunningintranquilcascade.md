# MTG Toolkit: Vite + React + FastAPI + SQLite

## Context

The existing toolkit is a Jupyter-style Python codebase for analysing sealed/draft MTG card pools. Text files in `card data/` are the current source of truth. The goal is to replace that with:
- A **SQLite database** as the single source of truth for cards and collections
- A **FastAPI backend** exposing a JSON API that wraps the existing fetch/encode/plot logic
- A **Vite + React frontend** with Plotly.js charts and the four core workflows

MTGA database integration stays as-is (low priority, not in scope).

---

## Architecture

```
┌────────────────────┐     HTTP/JSON      ┌────────────────────────┐
│   Vite + React     │ ←────────────────→ │  FastAPI (main.py)     │
│   frontend/        │                    │  backend/              │
│                    │                    │    ├─ database.py       │
│  Plotly.js renders │                    │    └─ routers/         │
│  fig JSON from API │                    │       ├─ collections.py │
└────────────────────┘                    │       └─ analysis.py   │
                                          └──────────┬─────────────┘
                                                     │
                              ┌──────────────────────┼───────────────┐
                              │                      │               │
                         SQLite DB            Scryfall API     mtg_func.py
                         (cards,              (fetch only)     mtg_plot.py
                          collections)                         mtg_keywords.py
                                                               constants.py
```

---

## Step 1 — Refactor `mtg_plot.py` and `mtg.py`

**`mtg_plot.py`** — All plot functions must return `go.Figure` objects, not call `.show()`.

| Function | Change |
|---|---|
| `plot_simple_bar()` | `return fig.show()` → `return fig` |
| `blb_plots()` | Collect both `plot_simple_bar()` returns → `return [fig1, fig2]` |
| `otj_plots()` | Collect all three figs → `return [fig1, fig2, fig3]` |
| `mh3_plots()` | Collect the single fig → `return [fig]` |
| `plot_set_specific()` | Propagate the list returned by the dispatched function → `return figs or []` |

**`mtg.py`** — `visualize_deck()` currently calls each plot inline. Refactor to:
- Accumulate all figures into a `list`
- Return the list (existing notebook-cell call sites just ignore the return value — no breakage)
- Remove all `.show()` calls within the function

---

## Step 2 — Database schema (`backend/database.py`)

Three tables in a local `mtg.db` SQLite file:

```sql
CREATE TABLE IF NOT EXISTS cards (
    id            INTEGER PRIMARY KEY,
    collector_no  TEXT NOT NULL,
    set_id        TEXT NOT NULL,
    name          TEXT,
    mana_cost     TEXT,
    cmc           REAL,
    power         TEXT,
    toughness     TEXT,
    type_line     TEXT,
    oracle_text   TEXT,
    rarity        TEXT,
    colours       TEXT,   -- JSON array e.g. '["B","W"]'
    keywords      TEXT,   -- JSON array
    price         REAL,
    price_std     REAL,
    price_foil    REAL,
    price_etched  REAL,
    UNIQUE(collector_no, set_id)
);

CREATE TABLE IF NOT EXISTS collections (
    id         INTEGER PRIMARY KEY,
    name       TEXT UNIQUE NOT NULL,
    set_id     TEXT,
    type       TEXT,    -- 'pool' | 'deck' | 'sideboard' | 'commander'
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS collection_cards (
    id            INTEGER PRIMARY KEY,
    collection_id INTEGER REFERENCES collections(id) ON DELETE CASCADE,
    card_id       INTEGER REFERENCES cards(id),
    quantity      INTEGER DEFAULT 1,
    foil          INTEGER DEFAULT 0,
    etched        INTEGER DEFAULT 0
);
```

`database.py` exposes `get_db()` (yields a `sqlite3.Connection`) and `init_db()` (creates tables on startup). **No SQLAlchemy** — plain `sqlite3`.

---

## Step 3 — FastAPI backend (`backend/main.py` + routers)

`main.py` mounts two routers and calls `init_db()` on startup. CORS is enabled for `localhost:5173` (Vite dev server).

### `routers/collections.py`

| Method | Path | Action |
|---|---|---|
| `GET` | `/api/collections` | List all collections (id, name, type, set_id, card count) |
| `POST` | `/api/collections` | Create collection `{name, set_id, type}` |
| `DELETE` | `/api/collections/{id}` | Delete collection + its card links |
| `GET` | `/api/collections/{id}/cards` | Return cards in collection as JSON rows |
| `POST` | `/api/collections/{id}/ingest/file` | Upload text file (collector numbers) + set_id; fetch from Scryfall via `get_card_info()`, upsert into `cards`, link to collection |
| `GET` | `/api/collections/{id}/ingest/set?set_id=xxx` | **SSE stream** — fetches full set from Scryfall card-by-card; emits `data: {"done": N, "total": M, "name": "..."}` events, then a final `data: {"complete": true}` event. Uses FastAPI `StreamingResponse` with `text/event-stream`. Frontend opens an `EventSource` and updates a progress bar in real time. |
| `POST` | `/api/collections/move` | Body `{from_id, to_id, card_ids: []}` — moves `collection_cards` rows |

Ingest re-uses **existing** `get_card_info()` and `loop_cards()` from `mtg_func.py` unchanged. After fetching, card data is written to `cards` table (upsert on `collector_no + set_id`) and a `collection_cards` row is created.

### `routers/analysis.py`

| Method | Path | Action |
|---|---|---|
| `GET` | `/api/collections/{id}/analysis` | Returns `{"figures": [...]}` — list of Plotly JSON dicts |

Flow:
1. Load collection's cards from SQLite → `pd.DataFrame` matching `Constants.col`
2. `json.loads()` the stored colour/keyword JSON columns back to lists
3. `encode_features(df)` (unchanged from `mtg_func.py`)
4. `visualize_deck(df, set_id)` → list of `go.Figure`
5. Return `[fig.to_dict() for fig in figs]`

---

## Step 4 — Vite + React frontend (`frontend/`)

```
frontend/
  index.html
  vite.config.js     # proxy /api → localhost:8000
  package.json       # react, react-dom, react-plotly.js, plotly.js
  src/
    main.jsx
    App.jsx
    api.js           # fetch wrappers for all API routes
    components/
      Sidebar.jsx    # action radio, collection selector
      PoolAnalysis.jsx
      IngestCards.jsx
      BuildDeck.jsx
      CommanderDeck.jsx
```

`vite.config.js` proxies `/api` to `localhost:8000` so the frontend needs no base URL config.

### Component responsibilities

**`Sidebar.jsx`**: action radio (`Sealed/Draft Analysis` | `Ingest Cards` | `Build Deck` | `Commander Deck`), collection dropdown (fetched from `GET /api/collections`), "New Collection" button.

**`PoolAnalysis.jsx`**: On collection select → `GET /api/collections/{id}/analysis` → renders each figure dict with `<Plot data={fig.data} layout={fig.layout} />` from `react-plotly.js`. Also shows a summary card table above the charts.

**`IngestCards.jsx`**: Set ID input + file upload (`<input type="file">`) or "Full set" toggle.
- File upload: `POST /api/collections/{id}/ingest/file` (fast, one HTTP call)
- Full set: opens an `EventSource` to `GET /api/collections/{id}/ingest/set?set_id=xxx`; renders a progress bar (`done / total`) updating in real time as each SSE event arrives; shows card name currently being fetched; closes `EventSource` on `complete` event.

**`BuildDeck.jsx`**: Loads collection cards via `GET /api/collections/{id}/cards`. Renders a table with a checkbox column. Running count "X / 40 selected". "Save as Deck" creates a new collection via `POST /api/collections` then `POST /api/collections/move` for selected cards.

**`CommanderDeck.jsx`**: Same as `PoolAnalysis.jsx` — reuses the same analysis endpoint.

---

## Migration of existing text files (one-time)

A `migrate.py` script at the repo root reads each file in `card data/`, detects set from filename (`YYMMDD_SETID_TYPE.txt`), calls `loop_cards()`, and writes to the database. Run once manually — not part of the app.

---

## Files created / modified

| File | Status | Change |
|---|---|---|
| `mtg_plot.py` | Modified | All plot functions return `fig` / `[figs]` |
| `mtg.py` | Modified | `visualize_deck()` returns list of figures |
| `backend/__init__.py` | New | Empty |
| `backend/main.py` | New | FastAPI app, CORS, startup |
| `backend/database.py` | New | `init_db()`, `get_db()` using `sqlite3` |
| `backend/routers/collections.py` | New | Collection + ingest endpoints |
| `backend/routers/analysis.py` | New | Analysis endpoint |
| `frontend/` | New | Vite + React app (see tree above) |
| `migrate.py` | New | One-time import of `card data/` text files |

`mtga-mtg_card.py` SQLAlchemy → sqlite3 deferred (still low priority).

---

## Verification

1. `cd backend && uvicorn main:app --reload` — API available at `localhost:8000/docs`
2. `cd frontend && npm run dev` — app at `localhost:5173`
3. Create a collection, ingest `card data/240607_mh3_sealed_cards.txt` via the UI
4. Switch to Pool Analysis — Plotly charts render in browser
5. Switch to Build Deck — select 40 cards, save, confirm new deck collection appears in sidebar
6. `migrate.py` — run and confirm all existing `card data/` files appear as collections
