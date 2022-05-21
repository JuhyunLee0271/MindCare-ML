import pymysql

# RDS CONFIG
HOST = 'flask-db.cuqw33e66jfm.ap-northeast-2.rds.amazonaws.com'
USER = 'admin'
PW = 'capstoneml'
DB = 'recommendation'
CHARSET = 'utf8'
PORT = 3306

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
    query = "SELECT title, artist FROM MUSIC WHERE musicId = %s"
    param = (musicId)
    cur.execute(query, param)
    result = cur.fetchall()
    if not result:
        return ""
    
    disconnect_from_db(conn, cur)
    return {'title': result[0][0], 'artist': result[0][1]}

if __name__ == "__main__":
    conn, cur = connect_to_db()
    print(lookup_music_info(174749))
    disconnect_from_db(conn, cur)
