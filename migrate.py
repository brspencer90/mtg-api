"""
One-time migration: imports all card data/*.txt files into the SQLite database.

Run from the project root:
    python migrate.py
"""
from db.migrations.import_legacy import main

if __name__ == '__main__':
    main()
