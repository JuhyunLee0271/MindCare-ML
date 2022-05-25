from recommender import recommender
from extractor import extractor

# 추천을 위한 API 호출을 담당하는 메인 컨트롤러
class main_controller:
    
    def __init__(self):
        self.extractor = extractor()        # 키워드, 감정 추출 
        self.recommender = recommender()    # 음악, 행동, 음식 추천
        self.diary = None
                
    # load user's diary(content) for recommendation
    def get_diary(self, content):
        self.diary = content
        
    def keyword_extract(self):
        keywords = []
        if self.diary:
            keywords = self.extractor.extract_keyword_from_diary(self.diary)
        else: print("There is no diary contents!")
        
        return keywords if keywords else []
        
    def sentiment_extract(self):
        emotion = None
        if self.diary:
            emotion = self.extractor.extract_sentiment_from_diary(self.diary)
        else: print("There is no diary contents!")
        
        return emotion if emotion else ""
     
    def music_recommend(self, **kwargs):
        musics = []
        # with kwargs, divide weather/time recommendation and emotion/keywords recommendation
        if 'weather' in kwargs and 'time' in kwargs:
            musics = self.recommender.recommend_music_with_tags(
                weather = kwargs['weather'], time = kwargs['time'])
        
        # keyword is optional parameter
        elif 'emotion' in kwargs:
            musics = self.recommender.recommend_music_with_tags(
                emotion = kwargs['emotion'], keywords = kwargs['keywords']
            )
            
        else: print("There are no weather/time or emotion/keywords!")
            
        return musics if musics else "Music: There are no recommended musics!"
        
    def food_recommend(self, **kwargs):
        foods = []
        if 'emotion' in kwargs:
            foods = self.recommender.recommend_food_with_emotion(kwargs['emotion'])
        else: print("There is no emotion!")
        
        return foods if foods else "Foods: There are no recommended foods"

    def behavior_recommend(self, **kwargs):
        behaviors = []
        if 'emotion' in kwargs:
            behaviors = self.recommender.recommend_behavior_with_emotion(kwargs['emotion'])
        else: print("There is no emotion!")
        
        return behaviors if behaviors else "Behavior: There are no recommended behaviors!"