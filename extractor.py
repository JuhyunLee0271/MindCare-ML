"""
CLASS
extractor.py
사용자의 일기로부터 감정, 키워드를 추출하는 class

METHODS
    
    - extract_sentiment_from_diary(content: str) -> str:
        return emotion of user
        e.g.,
        {'emotion': '슬픔'}
    
    - extract_keyword_from_diary(content: str) -> list:
        return list of keywords
        e.g.,
        [
            {'keyword1': '강남'},
            {'keyword2': '토끼정'},
            {'keyword3': '쉑쉑버거'}
        ]
"""

# requirements
from krwordrank.word import summarize_with_keywords
from krwordrank.hangle import normalize
from konlpy.tag import Okt
import kss

class extractor:
    def __init__(self):
        self.keyword_extractor = Keyword_Extractor()
        self.kobert_model = KoBERT_model()
    
    # extract emotion from content(diary)
    def extract_sentiment_from_diary(self, content):
        emotion = "기쁨"
        # emotion = self.kobert_model.run()
        return emotion 
    
    # extract keywords from content(diary)
    def extract_keyword_from_diary(self, content):
        keywords = self.keyword_extractor.run(content)
        return keywords

class Keyword_Extractor:
    def __init__(self, PATH = '../keyword_extract/stopwords-ko.txt', 
                JVM_PATH = '/Library/Java/JavaVirtualMachines/zulu-11.jdk/Contents/Home/bin/java'):
        self.path = PATH
        self.okt = Okt(jvmpath=JVM_PATH)
        # load saved korean stopwords
        with open(self.path, 'r') as f:
            self.stopwords = [word.rstrip('\n') for word in f.readlines()]
            
    def run(self, text: str):
        try:
            sentences = [normalize(sentence, english = True, number = True).replace('.', '') 
                        for sentence in kss.split_sentences(text)]
            
            keywords = summarize_with_keywords(
                sentences, min_count = 2, max_length = 10,
                beta = 0.85, max_iter = 10, stopwords = self.stopwords, verbose = False
            )
            return [tag for tag in self.okt.nouns(' '.join(keywords)) if tag not in self.stopwords][:3]
        except:
            return []
        
class KoBERT_model: