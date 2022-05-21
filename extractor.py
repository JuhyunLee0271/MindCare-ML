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

# requirements for keyword extract
from krwordrank.word import summarize_with_keywords
from krwordrank.hangle import normalize
from konlpy.tag import Okt
import kss
import numpy as np

# requirements for KoBERT
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

class extractor:
    def __init__(self):
        self.keyword_extractor = Keyword_Extractor()
        self.sentiment_extractor = Sentiment_Extractor()
    
    # extract emotion from content(diary)
    def extract_sentiment_from_diary(self, content):
        emotion = self.sentiment_extractor.run(content)
        return emotion
    
    # extract keywords from content(diary)
    def extract_keyword_from_diary(self, content):
        keywords = self.keyword_extractor.run(content)
        return keywords

class Keyword_Extractor:
    def __init__(self, PATH = './requirements/stopwords-ko.txt'):
        self.path = PATH
        self.okt = Okt()

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

class Sentiment_Extractor:
    def __init__(self):
        # Load CUDA and default bertmodel, vocabulary 
        self.device = torch.device("cuda:0")
        self.bertmodel, self.vocab = get_pytorch_kobert_model()
        
        self.emo_dict = {0: '중립', 1: '걱정', 2: '슬픔', 3: '분노', 4: '행복'}
        
        # Setting Hyper-parameters
        self.max_len = 64
        self.batch_size = 32
        self.warmup_ratio = 0.1
        self.num_epochs = 20
        self.max_grad_norm = 1
        self.log_interval = 100
        self.learning_rate = 10e-5
        
        # Load pretrained model
        self.model = torch.load('./model/trained_kobert.pt')
        self.model.load_state_dict(torch.load('./model/model_state_dict.pt'))

        # Load tokenizer
        self.tokenizer = get_tokenizer()
        self.tok = nlp.data.BERTSPTokenizer(self.tokenizer, self.vocab, lower=False)
        
    # Define softmax Function
    def new_softmax(self, a):
        c = np.max(a)
        exp_a = np.exp(a-c)
        sum_exp_a = np.sum(exp_a)
        y = (exp_a / sum_exp_a) * 100
        return np.round(y, 3)
    
    # Return Prediction of Emotion for diary content 
    def run(self, sentence):
        data = [sentence, '0']
        dataset_another = [data]
        
        another_test = BERTDataset(dataset_another, 0, 1, self.tok, self.max_len, True, False)
        test_dataloader = torch.utils.data.DataLoader(
                        another_test, batch_size=self.batch_size, num_workers=5)
        
        self.model.eval()
        
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
            token_ids = token_ids.long().to(self.device)
            segment_ids = segment_ids.long().to(self.device)
            
            valid_length = valid_length
            label = label.long().to(self.device)
            
            out = self.model(token_ids, valid_length, segment_ids)
            for i in out:
                logits = i
                logits = logits.detach().cpu().numpy()
                probability = []
                logits = np.round(self.new_softmax(logits), 3).tolist()
                for logit in logits:
                    probability.append(np.round(logit, 3))
                emotion = self.emo_dict[np.argmax(logits)]
            return emotion
        
"""
Helper Class for KoBERT
"""
# Define Model for KoBERT Classifier
class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=5,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)
    
# Input Dataset for KoBERT
class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels)) 

if __name__ == "__main__":
    kobert = Sentiment_Extractor()
    text = "오늘 날씨도 너무 좋고 하던 일도 잘 풀려서 기분이 좋았어"
    print(kobert.run(text))