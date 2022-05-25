"""
CLASS
recommender.py
extractor로 부터 추출한 감정과 키워드를 기반으로 음악, 행동, 음식을 추천하는 class

METHODS

    - recommend_music_with_tags(emotion: str, *args: list) -> list:
        return list of 20 musics
        e.g.,
        [
            {'Title': '밤 편지', 'Artist': '아이유'}, 
            {'Title': '가나다라', 'Artist': 박재범},
            ... 
        ]
        
    - recommend_food_with_emotion(emotion: str) -> list:
        return list of 3 foods
        e.g.,
        [
            {'food1': '엽떡'},
            {'food2': '제육'},
            {'food3': '고추바사삭'}
        ]
        
    - recommend_behavior_with_emotion(emotion: str) -> list:
        return list of 2 behaviors
        e.g.,
        [
            {'behavior1': '산책하기'},
            {'behavior2': '영화보기'}
        ]
    
"""
import random
from dp_api import connect_to_db, disconnect_from_db
import numpy as np
from gensim.models import FastText

class recommender:
    def __init__(self):
        self.music_recommender = Music_recommender()
        self.food_recommender = Food_recommender()
        self.behavior_recommender = Behavior_recommender()
    
    # recommend 20 musics from user's emotion and tags
    def recommend_music_with_tags(self, **kwargs):
        if 'weather' in kwargs and 'time' in kwargs:
            recom_musics = self.music_recommender.run(
                weather = kwargs['weather'], time = kwargs['time'])
        elif 'emotion' in kwargs:
            recom_musics = self.music_recommender.run(
                emotion = kwargs['emotion'], keywords = kwargs['keywords'])
        else:
            print("Invalid parameters passed(weather/time or emotion/keywords)")
            return None
        return recom_musics

    # recommend 3 foods from user's emotion
    def recommend_food_with_emotion(self, emotion: str):
        recom_foods = self.food_recommender.run(emotion)
        return recom_foods

    # recommend 2 behaviors from user's emotion
    def recommend_behavior_with_emotion(self, emotion: str):
        recom_bahaviors = self.behavior_recommender.run(emotion)
        return recom_bahaviors
    
class Music_recommender:
    def __init__(self):
        self.model = FastText.load('./model/trained_fasttext.model')
    
    # load each cluster's tags 
    def load_cluster_tags(self):
        conn, cur = connect_to_db()
        cur.execute("SELECT COUNT(*)/10 FROM CLUSTER")
        self.n_clusters = int(cur.fetchall()[0][0])

        cluster_tags = []
        try:
            query = "SELECT tag FROM CLUSTER WHERE label = %s"
            for i in range(self.n_clusters):
                param = (i)
                cur.execute(query, param)
                cluster_tags.append([tag for tag, in cur.fetchall()])
        except Exception as e:
            print(e)
            pass
        disconnect_from_db(conn, cur)
        return cluster_tags
    
    # find most similar cluster with keywords
    def find_similar_clusters(self, keywords):
        cluster_tags = self.load_cluster_tags()
        similarities = []

        for cluster_tag in cluster_tags:
            sim = 0
            for tag in cluster_tag:
                for keyword in keywords:
                    sim += self.model.wv.similarity(keyword, tag)
            similarities.append(sim)
        return np.argsort(similarities)[::-1][:self.n_clusters//5]

    # recommend 20 musics randomly with weights
    def run(self, **kwargs):
        isWeather, isNeutral = False, False
        isWeather = False
        if 'weather' in kwargs and 'time' in kwargs:
            keywords = [kwargs['weather'], kwargs['time']]
            sim_clusters = self.find_similar_clusters(keywords)
            isWeather = True
        
        else:
            if kwargs['emotion'] == '중립':
                keywords = ['기쁨', *kwargs['keywords'], '슬픔']
            
            keywords = [kwargs['emotion'], *kwargs['keywords']]
            sim_clusters = self.find_similar_clusters(keywords[0])
        
        conn, cur = connect_to_db()

        # load similar cluster's musics
        target_music = []
        query = "SELECT musicId, tag, cnt FROM MUSIC WHERE label = %s"
        for label in sim_clusters:
            param = (label)
            cur.execute(query, param)
            target_music.extend(cur.fetchall())
        
        # calculate similarities each musics
        res = []
        
        # weather/time 
        if not flag:
            for musicId, tags, cnt in target_music:
                sim = 0
                for tag in tags.split():
                    for keyword in keywords:
                        sim += self.model.wv.similarity(tag, keyword)
                cnt_weight = np.log(cnt)
                res.append([musicId, 10*sim/(len(tags)*len(keywords)) + cnt_weight])
    
        # emotion/keywords 
        else:
            for musicId, tags, cnt in target_music:
                sim = 0
                for tag in tags.split():
                    for keyword in keywords:
                        if keyword == kwargs['emotion']: sim += 10*self.model.wv.similarity(tag, keyword)
                        else: sim += self.model.wv.similarity(tag, keyword)
                cnt_weight = np.log(cnt)
                res.append([musicId, 10*sim/(len(tags)*len(keywords)) + cnt_weight])

        res.sort(key=lambda x:-x[1])
        
        result = [musicId for musicId, sim in res[:10]]
        
        musicId = [musicId for musicId, sim in res[10:]]
        weights = [sim for musicId, sim in res[10:]]
        weights = [sim / sum(weights) for sim in weights]

        result.extend(np.random.choice(
            musicId, p = weights, size = 10
        ))
        
        random.shuffle(result)
                
        recom_musics = []
        query = "SELECT title, artist FROM MUSIC WHERE musicId = %s"
        for m_id in result:
            param = (m_id)
            cur.execute(query, param)
            recom_musics.append(cur.fetchall()[0])
        
        disconnect_from_db(conn, cur)
        return recom_musics

class Food_recommender:
    def __init__(self):
        pass
    
    def run(self, emotion: str):
        conn, cur = connect_to_db()
        
        """
        Food Recommendation Logic
        """
        
        disconnect_from_db(conn, cur)
        return ['엽떡', '제육', '고추바사삭']

class Behavior_recommender:
    def __init__(self):
        self.emo_dict = {'중립': 0, '걱정': 1, '슬픔': 2, '분노': 3, '행복': 4}
        
    def run(self, emotion: str):
        conn, cur = connect_to_db()
        
        query = "SELECT `name`, content FROM BEHAVIOR WHERE label = %s"
        param = (self.emo_dict[emotion])
        cur.execute(query, param)
        result = cur.fetchall()
        if not result:
            return ""
        
        disconnect_from_db(conn, cur)
        return result[random.randint(0, len(result)-1)]

if __name__ == "__main__":
    recom = Music_recommender()
    print(recom.run(emotion = '행복', keywords = ['사랑']))
    # print(recom.run(emotion = '슬픔', keywords = ['비', '새벽']))
    
    
    