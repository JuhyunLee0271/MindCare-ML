from flask import Flask, jsonify, request
from flask_cors import CORS
from dp_api import lookup_music_info
from controller import main_controller

app = Flask(__name__)
CORS(app)
controller = main_controller()

# 마이페이지에서 music의 정보를 넘기는 API
@app.route('/mypage/music', methods=["GET"])
def request_music_info():
    musicId = request.args.get('musicId')
    musicInfo = lookup_music_info(musicId)

    return jsonify(
        title = musicInfo['title'],
        artist = musicInfo['artist']
    ) if musicInfo else "No song Info corresponding to that musicId!"

# 날씨/시간으로 음악 추천 (일기 쓰기 전에 첫 페이지)
@app.route('/music/weather', methods=["GET"])
def weather_recommendation():
    time_convert = {0: '새벽', 1: '아침', 2: '낮', 3: '밤'}
    weather = request.args.get('weather')
    time = request.args.get('time')

    if not weather or not time:
        return 'Bad Request!'

    time = time_convert[int(time.split(':')[0])//6]
    
    music_list = []
    music_list = controller.music_recommend(
        weather = weather,
        time = time
    )

    return jsonify(
        musicList = music_list
    ) if music_list else "There is no Recommendation corresponding to that weather/time"

# 감정, 키워드로 음악/행동/음식 추천(일기)
@app.route('/music/diary', methods=["GET"])
def diary_recommendation():
    content = request.args.get('content')
    if not content:
        return 'Bad Request!'
    
    music_list, food_list, behavior_list = [], [], []
    
    # Get user's diary content and extract emotion/keywords from that
    controller.get_diary(content)
    emotion = controller.sentiment_extract()
    keywords = controller.keyword_extract()
    
    if emotion:
        # Recommend music/food/behavior with emotion/keywords
        music_list = controller.music_recommend(emotion = emotion, keywords = keywords)
        food_list = controller.food_recommend(emotion = emotion)
        behavior_list = controller.behavior_recommend(emotion = emotion)
    
    return jsonify(
        musicList = music_list, 
        foodList = food_list, 
        behaviorList = behavior_list,
        keywordList = keywords
        ) if music_list and food_list and behavior_list else "There is no Recommendation corresponding to that emotion!"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)