import json
import sqlite3
import uuid

from .data import connect

ORDERS = [
    ("UC882140", "DEMO-01", "In transit", "Expected delivery 20 November 2026"),
    ("UC882141", "DEMO-01", "Delivered", "Marked delivered 13 November 2026"),
    ("UC882142", "DEMO-02", "Packed", "Dispatch is pending"),
]


class Store:
    def __init__(self, path, products):
        self.path = path
        with connect(self.path) as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS orders (
                    id TEXT PRIMARY KEY, customer TEXT, status TEXT, detail TEXT);
                CREATE TABLE IF NOT EXISTS tickets (
                    id TEXT PRIMARY KEY, turn_key TEXT UNIQUE, reason TEXT, summary TEXT);
                CREATE TABLE IF NOT EXISTS session (
                    id TEXT PRIMARY KEY, context TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS catalog (
                    sku TEXT PRIMARY KEY, name TEXT, category TEXT, price INTEGER, stock INTEGER, rating REAL);
                CREATE INDEX IF NOT EXISTS catalog_filters ON catalog(category, price, stock);
            """)
            db.executemany("INSERT OR IGNORE INTO orders VALUES (?,?,?,?)", ORDERS)
            db.execute("DELETE FROM catalog")
            db.executemany("INSERT INTO catalog VALUES (?,?,?,?,?,?)",
                          [(p["sku"], p["name"], p["category"], p["price_paise"], p["stock"], p["rating"])
                           for p in products.values()])

    def session(self, session_id):
        with connect(self.path) as db:
            row = db.execute("SELECT context FROM session WHERE id=?", (session_id,)).fetchone()
        return json.loads(row[0]) if row else {}

    def save_session(self, session_id, context):
        with connect(self.path) as db:
            db.execute("INSERT OR REPLACE INTO session VALUES (?,?)", (session_id, json.dumps(context)))

    def order(self, order_id, customer):
        with connect(self.path) as db:
            row = db.execute("SELECT id,status,detail FROM orders WHERE id=? AND customer=?",
                             (order_id, customer)).fetchone()
        return dict(row) if row else None

    def candidates(self, *, name=None, category=None, maximum=None, minimum=None, stock=False,
                   exclude="", limit=5):
        where, values = ["sku != ?"], [exclude]
        if name:
            where.append("name = ?")
            values.append(name)
        if category:
            where.append("category = ?")
            values.append(category)
        if maximum is not None:
            where.append("price <= ?")
            values.append(maximum)
        if minimum is not None:
            where.append("price >= ?")
            values.append(minimum)
        if stock:
            where.append("stock > 0")
        with connect(self.path) as db:
            rows = db.execute(
                f"SELECT sku FROM catalog WHERE {' AND '.join(where)} "
                "ORDER BY rating IS NULL, rating DESC, price IS NULL, price ASC, sku LIMIT ?",
                [*values, limit],
            ).fetchall()
        return [row[0] for row in rows]

    def ticket(self, turn_key, reason, summary):
        ticket_id = f"DEMO-{uuid.uuid4().hex[:10].upper()}"
        with connect(self.path) as db:
            db.execute("INSERT OR IGNORE INTO tickets VALUES (?,?,?,?)",
                       (ticket_id, turn_key, reason, summary[:500]))
            return db.execute("SELECT id FROM tickets WHERE turn_key=?", (turn_key,)).fetchone()[0]
