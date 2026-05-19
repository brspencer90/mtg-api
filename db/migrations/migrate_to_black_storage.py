"""
One-time migration: move every owned copy to 'Black Storage Box' and
store the old location name as the purchase_source (if not already set).
"""
import sqlite3
from collections import Counter
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / 'mtg.db'


def run():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA foreign_keys = ON')

    bsb = conn.execute("SELECT id FROM locations WHERE name = 'Black Storage Box'").fetchone()
    if not bsb:
        cur = conn.execute("INSERT INTO locations (name, type) VALUES ('Black Storage Box', 'storage')")
        bsb_id = cur.lastrowid
        print(f"Created 'Black Storage Box' (id={bsb_id})")
    else:
        bsb_id = bsb['id']
        print(f"Found 'Black Storage Box' (id={bsb_id})")

    rows = conn.execute("""
        SELECT cp.copy_id, cp.location_id, l.name AS loc_name, oc.purchase_source
        FROM card_placements cp
        JOIN locations l ON l.id = cp.location_id
        JOIN owned_copies oc ON oc.id = cp.copy_id
        WHERE cp.location_id != ?
    """, (bsb_id,)).fetchall()

    if not rows:
        print("All cards are already in 'Black Storage Box'. Nothing to do.")
        conn.close()
        return

    print(f"\n{len(rows)} cards to move:\n")
    for name, count in sorted(Counter(r['loc_name'] for r in rows).items(), key=lambda x: -x[1]):
        print(f"  {count:4}  {name}")

    import sys
    if '--yes' not in sys.argv:
        ans = input(f"\nProceed? [y/N] ")
        if ans.strip().lower() != 'y':
            print("Aborted.")
            conn.close()
            return

    for row in rows:
        copy_id   = row['copy_id']
        from_id   = row['location_id']
        loc_name  = row['loc_name']

        if not row['purchase_source']:
            conn.execute(
                "UPDATE owned_copies SET purchase_source = ? WHERE id = ?",
                (loc_name, copy_id),
            )

        conn.execute(
            "UPDATE card_placements SET location_id = ?, placed_at = datetime('now') WHERE copy_id = ?",
            (bsb_id, copy_id),
        )

        conn.execute("""
            INSERT INTO card_movements (copy_id, from_location_id, to_location_id, reason)
            VALUES (?, ?, ?, 'Migrated: source locations moved to Black Storage Box')
        """, (copy_id, from_id, bsb_id))

    conn.commit()
    conn.close()
    print(f"\nDone — {len(rows)} cards moved to 'Black Storage Box'.")


if __name__ == '__main__':
    run()
