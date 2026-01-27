import pymysql
conn=pymysql.connect(host='localhost',user='root',password='',database='chat_bot')

def logindata(qry):
    cursor=conn.cursor()
    sql=qry
    cursor.execute(sql)
    data=cursor.fetchone()
    #data=cursor
    return data

def selectdata(qry):
    cursor=conn.cursor()
    sql=qry
    cursor.execute(sql)
    data=cursor.fetchone()
    #data=cursor
    return data

def insertdata(qry):        
    cursor=conn.cursor()
    sql=qry
    cursor.execute(sql)
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
    cursor=conn.cursor()
    sql=qry
    cursor.execute(sql)
    #data=cursor.fetchone()
    data=cursor
    return data

def updatedata(qry):        
    cursor=conn.cursor()
    sql=qry
    cursor.execute(sql)
    conn.commit()