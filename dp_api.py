import pymysql
HOST = 'flask-db.cuqw33e66jfm.ap-northeast-2.rds.amazonaws.com'
USER = 'admin'
PW = 'capstoneml'
DB = 'recommendation'
CHARSET = 'utf8'
PORT = 3306


"""
db_api.py
db에 접근하는 api들
"""
# RDS connection Method
def connect_to_db():
    conn, cur = None, None
    try:
        conn = pymysql.connect(
            host = HOST,
            user = USER,
            password = PW,
            db = DB,
            charset = CHARSET,
            port = PORT
        )
        cur = conn.cursor()
    except:
        pass
    return conn, cur

def disconnect_from_db(conn, cur):
    if not conn or not cur:
        return
    conn.commit()
    conn.close()
        
# musicId -> {'title': title, 'artist': artist}
def lookup_music_info(musicId):
    conn, cur = connect_to_db()
    
    cur.execute("SELECT title, artist FROM MUSIC WHERE musicId = ?", [musicId])
    result = cur.fetchall()
    if not result:
        return ""
    
    disconnect_from_db(conn, cur)
    return {'title': result[0][0], 'artist': result[0][1]}
