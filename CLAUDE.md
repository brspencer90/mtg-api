# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python toolkit for Magic: The Gathering card analysis. It fetches card data from the Scryfall API, encodes card attributes into a feature-rich pandas DataFrame, and generates Plotly visualizations for sealed/draft pool evaluation.

## Environment Setup

The virtual environment lives at `.env_mtg/`. Activate it before running any scripts:

```powershell
.env_mtg\Scripts\Activate.ps1
```

Install dependencies if needed:

```powershell
pip install pandas numpy plotly requests scikit-learn nltk sqlalchemy
```

NLTK stopwords must also be downloaded once:

```python
import nltk; nltk.download('stopwords'); nltk.download('punkt')
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
