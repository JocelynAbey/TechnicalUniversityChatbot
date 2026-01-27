import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "db.sqlite3"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    ensure_tables(conn)
    return conn


def ensure_tables(conn: sqlite3.Connection) -> None:
    init_path = Path(__file__).with_name("sqlite_init.sql")
    if not init_path.exists():
        return
    with init_path.open("r", encoding="utf-8") as f:
        sql = f.read()
    conn.executescript(sql)

def logindata(qry):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(qry)
        data = cursor.fetchone()
        return data

def selectdata(qry):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(qry)
        data = cursor.fetchone()
        return data

def insertdata(qry):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(qry)
        conn.commit()

def insertdata1(qry, params=None):        
    with get_connection() as conn:
        cursor = conn.cursor()
        try:
            if params:
                cursor.execute(qry, params)   # ✅ safe: escapes values automatically
            else:
                cursor.execute(qry)
            conn.commit()
        except Exception as e:
            conn.rollback()   # rollback if something goes wrong
            print("Insert Error:", e)
            raise
        finally:
            cursor.close()


def selectalldata(qry):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(qry)
    return cursor

def updatedata(qry):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(qry)
        conn.commit()