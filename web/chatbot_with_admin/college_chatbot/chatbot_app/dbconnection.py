import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "db.sqlite3"
conn = sqlite3.connect(DB_PATH)

def logindata(qry):
    cursor = conn.cursor()
    cursor.execute(qry)
    data = cursor.fetchone()
    return data

def selectdata(qry):
    cursor = conn.cursor()
    cursor.execute(qry)
    data = cursor.fetchone()
    return data

def insertdata(qry):
    cursor = conn.cursor()
    cursor.execute(qry)
    conn.commit()

def insertdata1(qry, params=None):        
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
    cursor = conn.cursor()
    cursor.execute(qry)
    data = cursor
    return data

def updatedata(qry):
    cursor = conn.cursor()
    cursor.execute(qry)
    conn.commit()