from flask import Flask, Response, request, jsonify
from playsound import playsound 
import cv2
import mediapipe as mp
import time
import datetime
import threading
import sqlite3
import pandas as pd
import joblib

import os
import json

import queue # TTS 작업 대기줄
from gtts import gTTS # --- [핵심 수정] gTTS import ---


from plyer import notification

# --- Flask 앱 및 상태 변수 초기화 ---
app = Flask(__name__)
lock = threading.Lock()
app_state = {'mode': 'idle', 'label': -1, 'collection_end_time': 0, 'sound_on': True, 'cooldown_seconds': 5, 'prediction_threshold': 0.5, 'desktop_alert_on': True}
model = None
features = ['right_ear_x', 'right_ear_y', 'right_shoulder_x', 'right_shoulder_y', 'shoulder_ear_diff_x']

# --- 새로운 기능: 다양한 자세 유형 감지 ---
posture_types = {
    0: 'good_posture', 1: 'forward_head', 2: 'rounded_shoulders',
    3: 'slouching', 4: 'shoulder_tilt', 5: 'head_tilt'
}

# 게임화 요소를 위한 전역 변수
user_stats = {
    'level': 1, 'experience': 0, 'current_streak': 0,
    'best_streak': 0, 'total_good_minutes': 0, 'badges': [], 'last_activity': None
}
last_seen_landmarks = None

# --- 통계 저장을 위한 전역 변수 ---
last_prediction = 0
good_time_today = 0
bad_time_today = 0

# --- 음성 피드백을 위한 전역 변수 ---
voice_feedback_enabled = True
tts_queue = queue.Queue() 
tts_lock = threading.Lock() # gTTS도 동시에 여러 파일 생성을 방지하기 위해 Lock 유지

# --- [핵심 수정] gTTS를 사용하는 작업자 스레드 ---
def tts_worker():
    """대기줄(Queue)에서 작업을 가져와 gTTS로 MP3를 만들고 playsound로 재생"""
    print("✅ TTS 엔진 작업자 스레드 시작됨 (gTTS 사용).")
    
    while True:
        try:
            text_to_speak = tts_queue.get() 
            if text_to_speak is None: break
                
            if voice_feedback_enabled:
                with tts_lock: # 한 번에 하나의 음성 파일만 생성 및 재생
                    try:
                        # 1. gTTS를 사용해 텍스트를 MP3 데이터로 변환
                        tts = gTTS(text=text_to_speak, lang='ko')
                        
                        # 2. MP3 데이터를 임시 파일로 저장
                        temp_filename = "temp_voice.mp3"
                        tts.save(temp_filename)
                        
                        # 3. playsound로 임시 파일 재생
                        playsound(temp_filename)
                        
                        # 4. 재생 후 임시 파일 삭제
                        os.remove(temp_filename) 

                    except Exception as e:
                        print(f"⚠️ TTS 재생 오류: {e}")
                        # 임시 파일이 남아있으면 삭제
                        if os.path.exists("temp_voice.mp3"):
                            os.remove("temp_voice.mp3")
                
            tts_queue.task_done()
        except Exception as e:
            print(f"⚠️ TTS 작업자 실행 중 오류: {e}")
            if 'tts_queue' in locals() or 'tts_queue' in globals():
                 tts_queue.task_done()

# --- [핵심 수정] speak_text 함수 (대기줄에 추가) ---
def speak_text(text):
    """텍스트를 음성 대기줄(Queue)에 추가"""
    global tts_queue, voice_feedback_enabled
    if not voice_feedback_enabled: return False
    try:
        tts_queue.put(text)
        return True
    except Exception as e:
        print(f"⚠️ 음성 대기줄 추가 오류: {e}")
        return False

# --- init_tts() 함수는 더 이상 필요 없음 ---

# --- AI 모델 로드 함수 ---
def load_model():
    global model
    if os.path.exists('posture_model.joblib'):
        model = joblib.load('posture_model.joblib')
        print("✅ AI 모델 로딩 완료.")
    else:
        model = None
        print("⚠️ 경고: 'posture_model.joblib' 모델 파일을 찾을 수 없습니다.")

# --- MediaPipe 초기화 (전역 변수) ---
mp_pose = mp.solutions.pose
pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- DB 관련 함수 ---
# [교체] init_db 함수 전체
def init_db():
    conn = sqlite3.connect('posture_data.db', check_same_thread=False)
    cursor = conn.cursor()
    
    # --- [유지] 현재 사용 중인 테이블 6개 ---
    cursor.execute(''' CREATE TABLE IF NOT EXISTS posture_log (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME NOT NULL, right_ear_x REAL, right_ear_y REAL, right_shoulder_x REAL, right_shoulder_y REAL, label INTEGER) ''')
    cursor.execute(''' CREATE TABLE IF NOT EXISTS daily_stats (id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT UNIQUE NOT NULL, good_seconds INTEGER NOT NULL, bad_seconds INTEGER NOT NULL) ''')
    cursor.execute(''' CREATE TABLE IF NOT EXISTS posture_types_log (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME NOT NULL, posture_type TEXT NOT NULL, confidence REAL NOT NULL, landmarks_data TEXT NOT NULL) ''')
    cursor.execute(''' CREATE TABLE IF NOT EXISTS user_gamification (id INTEGER PRIMARY KEY AUTOINCREMENT, user_level INTEGER DEFAULT 1, experience_points INTEGER DEFAULT 0, current_streak INTEGER DEFAULT 0, best_streak INTEGER DEFAULT 0, total_good_minutes INTEGER DEFAULT 0, badges TEXT DEFAULT '[]', last_activity DATETIME, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP) ''')
    cursor.execute(''' CREATE TABLE IF NOT EXISTS smart_notifications (id INTEGER PRIMARY KEY AUTOINCREMENT, notification_type TEXT NOT NULL, trigger_time TEXT NOT NULL, is_active BOOLEAN DEFAULT 1, last_triggered DATETIME, created_at DATETIME DEFAULT CURRENT_TIMESTAMP) ''')
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS goals (id INTEGER PRIMARY KEY AUTOINCREMENT, goal_type TEXT NOT NULL, target_value REAL NOT NULL, current_value REAL DEFAULT 0, start_date TEXT NOT NULL, end_date TEXT NOT NULL, is_active BOOLEAN DEFAULT 1, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    
    conn.commit()
    conn.close()


def save_to_db(landmarks, label):
    conn = sqlite3.connect('posture_data.db', check_same_thread=False)
    cursor = conn.cursor()
    data = (datetime.datetime.now(), landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y, label)
    cursor.execute('INSERT INTO posture_log (timestamp, right_ear_x, right_ear_y, right_shoulder_x, right_shoulder_y, label) VALUES (?, ?, ?, ?, ?, ?)', data)
    conn.commit()
    conn.close()

def detect_posture_type(landmarks):
    try:
        left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]; right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]; right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        
        posture_scores = {}
        shoulder_tilt_score = abs(left_shoulder.y - right_shoulder.y)
        posture_scores['shoulder_tilt'] = shoulder_tilt_score
        head_tilt_score = abs(left_ear.y - right_ear.y)
        posture_scores['head_tilt'] = head_tilt_score
        ear_avg_x = (left_ear.x + right_ear.x) / 2
        shoulder_avg_x = (left_shoulder.x + right_shoulder.x) / 2
        forward_head_score = abs(ear_avg_x - shoulder_avg_x)
        posture_scores['forward_head'] = forward_head_score
        posture_scores['rounded_shoulders'] = forward_head_score
        
        thresholds = {'shoulder_tilt': 0.04, 'head_tilt': 0.03, 'forward_head': 0.06, 'rounded_shoulders': 0.06}
        
        max_score = 0; detected_type = 'good_posture'
        for posture_type, score in posture_scores.items():
            if score > thresholds[posture_type] and score > max_score:
                max_score = score; detected_type = posture_type
        return detected_type, max_score
    except Exception as e:
        print(f"자세 유형 감지 오류: {e}"); return 'good_posture', 0.0

def save_posture_type_to_db(posture_type, confidence, landmarks):
    try:
        conn = sqlite3.connect('posture_data.db', check_same_thread=False); cursor = conn.cursor()
        landmarks_data = {
            'nose': {'x': landmarks[mp_pose.PoseLandmark.NOSE.value].x, 'y': landmarks[mp_pose.PoseLandmark.NOSE.value].y},
            'left_ear': {'x': landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x, 'y': landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y},
            'right_ear': {'x': landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x, 'y': landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y},
            'left_shoulder': {'x': landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 'y': landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y},
            'right_shoulder': {'x': landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 'y': landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y}
        }
        cursor.execute('''INSERT INTO posture_types_log (timestamp, posture_type, confidence, landmarks_data) VALUES (?, ?, ?, ?)''', 
                       (datetime.datetime.now(), posture_type, confidence, json.dumps(landmarks_data)))
        conn.commit(); conn.close()
    except Exception as e:
        print(f"자세 유형 저장 오류: {e}")

def get_posture_feedback_message(posture_type):
    feedback_messages = {
        'forward_head': "거북목 자세입니다. 턱을 뒤로 당기고 목을 곧게 펴주세요.",
        'rounded_shoulders': "어깨가 구부정합니다. 가슴을 펴고 어깨를 뒤로 젖혀주세요.",
        'slouching': "등이 구부정합니다. 허리를 곧게 펴고 앉아주세요.",
        'shoulder_tilt': "어깨가 기울어져 있습니다. 양 어깨를 균형있게 맞춰주세요.",
        'head_tilt': "머리가 기울어져 있습니다. 머리를 곧게 세워주세요.",
        'good_posture': "좋은 자세입니다! 계속 유지해주세요."
    }
    return feedback_messages.get(posture_type, "자세를 점검해주세요.")

def show_notification_if_enabled(message):
    """
    [신규 헬퍼 함수]
    알림을 실제로 띄우기 직전에, 토글이 여전히 켜져 있는지
    '다시 한 번' 확인합니다.
    """
    global lock, app_state
    
    # 2단계 확인: 알림 스레드가 실행되는 시점에도 토글이 켜져있는가?
    with lock:
        is_on = app_state.get('desktop_alert_on', True)
    
    if is_on: # 여전히 켜져 있을 때만 팝업을 띄웁니다.
        try:
            notification.notify(
                title='🚨 자세 경고! 🚨',
                message=message,
                app_name='AI 자세 교정 코치',
                timeout=5 
            )
        except Exception as e:
            print(f"🖥️ 데스크탑 알림 팝업 오류: {e}")

def load_user_gamification():
    global user_stats
    try:
        conn = sqlite3.connect('posture_data.db', check_same_thread=False); cursor = conn.cursor()
        cursor.execute('SELECT * FROM user_gamification ORDER BY id DESC LIMIT 1')
        result = cursor.fetchone()
        if result:
            user_stats = {
                'level': result[1], 'experience': result[2], 'current_streak': result[3],
                'best_streak': result[4], 'total_good_minutes': result[5],
                'badges': json.loads(result[6]) if result[6] else [], 'last_activity': result[7]
            }
        else:
            cursor.execute('''INSERT INTO user_gamification (user_level, experience_points, current_streak, best_streak, total_good_minutes, badges) 
                                VALUES (1, 0, 0, 0, 0, '[]')'''); conn.commit()
    

            user_stats = {
                'level': 1, 'experience': 0, 'current_streak': 0,
                'best_streak': 0, 'total_good_minutes': 0, 'badges': [], 'last_activity': None
            }

        conn.close()
        print(f"🎮 게임화 데이터 로드: 레벨 {user_stats['level']}, 경험치 {user_stats['experience']}")
    except Exception as e:
        print(f"게임화 데이터 로드 오류: {e}")

def update_user_gamification(posture_type, is_good_posture):
    global user_stats
    try:
        conn = sqlite3.connect('posture_data.db', check_same_thread=False); cursor = conn.cursor()
        if is_good_posture:
            exp_gain = 1; user_stats['experience'] += exp_gain; user_stats['current_streak'] += 1
            user_stats['total_good_minutes'] += 1
            new_level = (user_stats['experience'] // 100) + 1
            if new_level > user_stats['level']:
                user_stats['level'] = new_level; print(f"🎉 레벨업! 레벨 {user_stats['level']} 달성!")
            if user_stats['current_streak'] > user_stats['best_streak']:
                user_stats['best_streak'] = user_stats['current_streak']
            check_and_award_badges()
        else:
            user_stats['current_streak'] = 0
            
        cursor.execute('''UPDATE user_gamification SET 
                          user_level = ?, experience_points = ?, current_streak = ?, 
                          best_streak = ?, total_good_minutes = ?, badges = ?, 
                          last_activity = ?, updated_at = CURRENT_TIMESTAMP
                          WHERE id = 1''', 
                       (user_stats['level'], user_stats['experience'], user_stats['current_streak'],
                        user_stats['best_streak'], user_stats['total_good_minutes'], 
                        json.dumps(user_stats['badges']), datetime.datetime.now()))
        conn.commit(); conn.close()
    except Exception as e:
        print(f"게임화 데이터 업데이트 오류: {e}")

def check_and_award_badges():
    global user_stats
    badges_to_check = [
        {'name': 'first_steps', 'title': '첫 걸음', 'condition': lambda: user_stats['total_good_minutes'] >= 10, 'description': '좋은 자세 10분 달성'},
        {'name': 'streak_master', 'title': '연속 달인', 'condition': lambda: user_stats['current_streak'] >= 30, 'description': '30분 연속 좋은 자세'},
        {'name': 'posture_pro', 'title': '자세 전문가', 'condition': lambda: user_stats['total_good_minutes'] >= 100, 'description': '총 100분 좋은 자세'},
        {'name': 'level_up', 'title': '성장하는 자세', 'condition': lambda: user_stats['level'] >= 5, 'description': '레벨 5 달성'},
        {'name': 'perfect_day', 'title': '완벽한 하루', 'condition': lambda: user_stats['current_streak'] >= 480, 'description': '8시간 연속 좋은 자세'}
    ]
    for badge in badges_to_check:
        if badge['name'] not in user_stats['badges'] and badge['condition']():
            user_stats['badges'].append(badge['name'])
            print(f"🏆 배지 획득: {badge['title']} - {badge['description']}")

# --- [수정] analyze_posture_patterns가 날짜 범위를 인자로 받도록 변경 ---
def analyze_posture_patterns(start_date, end_date):
    """
    주어진 날짜 범위 내의 자세 패턴을 분석합니다.
    """
    try:
        conn = sqlite3.connect('posture_data.db', check_same_thread=False)
        cursor = conn.cursor()
        
        # [수정] SQL 쿼리가 날짜 범위를 사용하도록 변경
        # strftime('%H', timestamp)는 시간을 '00', '01'...'23' 형태의 문자열로 반환
        query = """
            SELECT strftime('%H', timestamp) as hour, posture_type, COUNT(*) as frequency 
            FROM posture_types_log 
            WHERE date(timestamp) BETWEEN ? AND ?
            AND posture_type != 'good_posture'
            GROUP BY hour, posture_type 
            ORDER BY frequency DESC
        """
        cursor.execute(query, (start_date, end_date))
        
        results = cursor.fetchall()
        hour_patterns = {}
        for hour_str, posture_type, frequency in results:
            hour = int(hour_str) # 문자열 '08' -> 숫자 8
            if hour not in hour_patterns: 
                hour_patterns[hour] = {}
            hour_patterns[hour][posture_type] = frequency
            
        worst_hours = []
        for hour, patterns in hour_patterns.items():
            total_bad = sum(patterns.values())
            worst_hours.append((hour, total_bad))
            
        worst_hours.sort(key=lambda x: x[1], reverse=True)
        conn.close()
        return worst_hours[:3] # 상위 3개 시간대 반환
        
    except Exception as e:
        print(f"자세 패턴 분석 오류: {e}")
        return []

# [교체] create_smart_notifications 함수
def create_smart_notifications():
    """
    [수정] 스마트 알림은 항상 '최근 7일' 기준으로 생성하도록 고정
    """
    try:
        # '최근 7일' 날짜 계산
        today = datetime.date.today()
        start_date = (today - datetime.timedelta(days=6)).isoformat()
        end_date = today.isoformat()
        
        # [수정] 7일 기준으로 헬퍼 함수 호출
        worst_hours = analyze_posture_patterns(start_date, end_date)
        
        conn = sqlite3.connect('posture_data.db', check_same_thread=False); cursor = conn.cursor()
        cursor.execute('DELETE FROM smart_notifications')
        for hour, frequency in worst_hours:
            alert_hour = hour - 1 if hour > 0 else 23
            cursor.execute('''INSERT INTO smart_notifications (notification_type, trigger_time, is_active) 
                               VALUES (?, ?, 1)''', ('posture_reminder', f"{alert_hour:02d}:00",))
        conn.commit(); conn.close()
        print(f"🔔 스마트 알림 생성 완료: {len(worst_hours)}개 알림 설정 (최근 7일 기준)")
    except Exception as e:
        print(f"스마트 알림 생성 오류: {e}")

def check_smart_notifications():
    try:
        current_time = datetime.datetime.now().strftime("%H:%M")
        conn = sqlite3.connect('posture_data.db', check_same_thread=False); cursor = conn.cursor()
        cursor.execute('''SELECT * FROM smart_notifications 
                          WHERE trigger_time = ? AND is_active = 1 
                          AND (last_triggered IS NULL OR last_triggered < datetime('now', '-1 hour'))''', (current_time,))
        notifications = cursor.fetchall()
        for notification in notifications:
            message = "자세 점검 시간입니다! 지금 자세를 확인해주세요."
            speak_text(message)
            cursor.execute('UPDATE smart_notifications SET last_triggered = CURRENT_TIMESTAMP WHERE id = ?', (notification[0],))
        conn.commit(); conn.close()
        return len(notifications) > 0
    except Exception as e:
        print(f"스마트 알림 체크 오류: {e}"); return False

def save_stats_to_db():
    today_str = datetime.date.today().isoformat()
    conn = sqlite3.connect('posture_data.db', check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute(''' INSERT INTO daily_stats (date, good_seconds, bad_seconds) VALUES (?, ?, ?) ON CONFLICT(date) DO UPDATE SET good_seconds = excluded.good_seconds, bad_seconds = excluded.bad_seconds; ''', (today_str, good_time_today, bad_time_today))
    conn.commit()
    conn.close()
    print(f"📊 오늘({today_str})의 통계 저장: 좋은 자세 {good_time_today}초, 나쁜 자세 {bad_time_today}초")

def load_stats_from_db():
    global good_time_today, bad_time_today
    today_str = datetime.date.today().isoformat()
    conn = sqlite3.connect('posture_data.db', check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('SELECT good_seconds, bad_seconds FROM daily_stats WHERE date = ?', (today_str,))
    result = cursor.fetchone()
    if result:
        good_time_today, bad_time_today = result
    else:
        good_time_today, bad_time_today = 0, 0
    conn.close()
    print(f"📊 오늘({today_str})의 통계 로드: 좋은 자세 {good_time_today}초, 나쁜 자세 {bad_time_today}초")

def track_posture_stats():
    global good_time_today, bad_time_today
    while True:
        with lock:
            if app_state['mode'] == 'coach':
                if last_prediction == 0: good_time_today += 1
                else: bad_time_today += 1
        time.sleep(1)

# --- 비디오 프레임 생성기 함수 ---
def generate_frames():
    global pose, mp_pose, mp_drawing, lock, model, features, last_prediction, app_state
    
    from plyer import notification # 함수 내에서 import
    
    cap = cv2.VideoCapture(0)
    last_alert_time = 0

    while True:
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        global last_seen_landmarks
        if results.pose_landmarks:
            last_seen_landmarks = results.pose_landmarks.landmark
        
        with lock:
            current_mode = app_state['mode']; end_time = app_state['collection_end_time']
            current_label = app_state['label']; is_sound_on = app_state['sound_on']
            current_threshold = app_state['prediction_threshold']
            cooldown_seconds = app_state['cooldown_seconds']
            is_desktop_alert_on = app_state.get('desktop_alert_on', True)

        skeleton_color = (128, 128, 128); status_text = "Idle"

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            if current_mode == 'coach' and model is not None:
                data = {'right_ear_x': landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x, 'right_ear_y': landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y, 'right_shoulder_x': landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 'right_shoulder_y': landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y}
                input_df = pd.DataFrame([data])
                input_df['shoulder_ear_diff_x'] = input_df['right_shoulder_x'] - input_df['right_ear_x']
                
                probabilities = model.predict_proba(input_df[features]) 
                proba_bad = probabilities[0][1]
                
                if proba_bad > current_threshold:
                    prediction_result = 1 # 나쁨
                else:
                    prediction_result = 0 # 좋음
                    
                last_prediction = prediction_result
                
                detected_posture_type, confidence = detect_posture_type(landmarks)
                is_good_posture = (detected_posture_type == 'good_posture')
                save_posture_type_to_db(detected_posture_type, confidence, landmarks)
                update_user_gamification(detected_posture_type, is_good_posture)
                
                if prediction_result == 0 and is_good_posture:
                    status_text = "Good Posture"; skeleton_color = (0, 255, 0)
                else:
                    if detected_posture_type == 'good_posture':
                        detected_posture_type = 'forward_head' 
                    
                    feedback_message = get_posture_feedback_message(detected_posture_type)
                    status_text = f"WARNING: {detected_posture_type.replace('_', ' ').title()}!"
                    skeleton_color = (0, 0, 255)

                    current_time = time.time()
                    if current_time - last_alert_time > cooldown_seconds:
                        if is_sound_on:
                            try:
                                threading.Thread(target=lambda: playsound('alert.mp3')).start()
                                speak_text(feedback_message)
                            except Exception as e: print(f"알림음 재생 오류: {e}")
                        
                        if is_desktop_alert_on:
                        # [수정] 람다(lambda) 대신, 새로 만든 헬퍼 함수를 스레드로 실행
                            try:
                                threading.Thread(target=show_notification_if_enabled, 
                                               args=(feedback_message,), 
                                                 daemon=True).start()
                            except Exception as e:
                                print(f"🖥️ 데스크탑 알림 스레드 생성 오류: {e}")
                        
                        last_alert_time = current_time
                
                if int(time.time()) % 60 == 0:
                    check_smart_notifications()
            
            elif current_mode == 'collect' and time.time() < end_time:
                save_to_db(landmarks, current_label)
                remaining_time = end_time - time.time()
                status_text = f"Collecting Label: {current_label} | {remaining_time:.1f}s"
                skeleton_color = (0, 255, 255)
            else:
                 status_text = "Ready"; skeleton_color = (0, 255, 0)

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, connection_drawing_spec=mp_drawing.DrawingSpec(color=skeleton_color, thickness=2))

        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, skeleton_color, 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- API 엔드포인트 ---
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_mode', methods=['POST'])
def set_mode():
    data = request.get_json()
    with lock:
        previous_mode = app_state['mode']
    mode = data.get('mode', 'idle'); label = data.get('label', -1); duration = data.get('duration', 0)
    with lock:
        app_state['mode'] = mode; app_state['label'] = label
        if mode == 'collect': app_state['collection_end_time'] = time.time() + duration
    if previous_mode == 'coach' and mode != 'coach': save_stats_to_db()
    print(f"Mode changed to: {mode}"); return jsonify({"status": "success", "mode": mode})

@app.route('/reload_model', methods=['POST'])
def reload_model():
    print("AI 모델을 다시 로드합니다..."); load_model()
    return jsonify({"status": "success", "message": "Model reloaded"})

@app.route('/clear_data', methods=['POST'])
# [교체] clear_data 함수 전체
@app.route('/clear_data', methods=['POST'])
def clear_data():
    try:
        conn = sqlite3.connect('posture_data.db', check_same_thread=False)
        cursor = conn.cursor()
        
        # --- [유지] 6개 테이블 초기화 ---
        cursor.execute('DELETE FROM posture_log'); 
        cursor.execute('DELETE FROM sqlite_sequence WHERE name="posture_log"')
        cursor.execute('DELETE FROM daily_stats'); 
        cursor.execute('DELETE FROM sqlite_sequence WHERE name="daily_stats"')
        cursor.execute('DELETE FROM posture_types_log'); 
        cursor.execute('DELETE FROM sqlite_sequence WHERE name="posture_types_log"')
        cursor.execute('DELETE FROM user_gamification'); 
        cursor.execute('DELETE FROM sqlite_sequence WHERE name="user_gamification"')
        cursor.execute('DELETE FROM smart_notifications'); 
        cursor.execute('DELETE FROM sqlite_sequence WHERE name="smart_notifications"')
        cursor.execute('DELETE FROM goals'); 
        cursor.execute('DELETE FROM sqlite_sequence WHERE name="goals"')
    
        
    
        
        conn.commit(); conn.close()
        
        global good_time_today, bad_time_today
        good_time_today, bad_time_today = 0, 0
        load_user_gamification() # 게임화 상태 초기화
        
        print("✅ 데이터베이스가 초기화되었습니다."); 
        return jsonify({"status": "success", "message": "Database cleared"})
        
    except Exception as e:
        print(f"⚠️ 데이터베이스 초기화 오류: {e}"); 
        return jsonify({"error": str(e)}), 500
        
@app.route('/set_sound', methods=['POST'])
def set_sound():
    data = request.get_json(); sound_status = data.get('sound_on', True)
    with lock:
        app_state['sound_on'] = sound_status
    print(f"🔊 Sound status changed to: {'ON' if sound_status else 'OFF'}"); return jsonify({"status": "success", "sound_on": sound_status})

@app.route('/set_cooldown', methods=['POST'])
def set_cooldown():
    data = request.get_json(); cooldown = data.get('cooldown', 5)
    cooldown = max(1, min(cooldown, 60)) 
    with lock:
        app_state['cooldown_seconds'] = cooldown
    print(f"⏱️ Alert cooldown changed to: {cooldown} seconds"); return jsonify({"status": "success", "cooldown": cooldown})

@app.route('/set_sensitivity', methods=['POST'])
def set_sensitivity():
    data = request.get_json(); threshold = data.get('threshold', 0.5)
    threshold = max(0.1, min(threshold, 0.9)) 
    with lock:
        app_state['prediction_threshold'] = threshold
    print(f"🧠 AI Prediction Threshold changed to: {threshold}"); return jsonify({"status": "success", "threshold": threshold})

@app.route('/voice_feedback', methods=['POST'])
def voice_feedback():
    data = request.get_json(); message = data.get('message', '')
    if not message: return jsonify({"status": "error", "message": "No message provided"}), 400
    success = speak_text(message)
    if success: print(f"🔊 Voice feedback: {message}"); return jsonify({"status": "success", "message": "Voice feedback sent"})
    else: return jsonify({"status": "error", "message": "Failed to play voice feedback"}), 500

@app.route('/set_voice_feedback', methods=['POST'])
def set_voice_feedback():
    data = request.get_json(); enabled = data.get('enabled', True)
    global voice_feedback_enabled
    voice_feedback_enabled = enabled
    print(f"🔊 Voice feedback {'enabled' if enabled else 'disabled'}")
    return jsonify({"status": "success", "enabled": enabled})

@app.route('/get_user_stats', methods=['GET'])
def get_user_stats():
    global user_stats
    return jsonify({"status": "success", "stats": user_stats})

# --- [신규 추가] Human-in-the-Loop 피드백 저장 API ---
@app.route('/save_feedback', methods=['POST'])
def save_feedback():
    global last_seen_landmarks
    data = request.get_json()
    label = data.get('label') # 사용자가 누른 버튼 (0: 좋음, 1: 나쁨)

    if label not in [0, 1]:
        return jsonify({"status": "error", "message": "Invalid label"}), 400
    
    try:
        with lock: # 랜드마크가 갱신되는 도중에 읽지 않도록 잠금
            if last_seen_landmarks is None:
                return jsonify({"status": "error", "message": "No landmarks captured yet"}), 400
            
            # (중요) 현재 랜드마크와 "사용자가 알려준 정답(label)"을 학습DB에 저장
            save_to_db(last_seen_landmarks, label) 
        
        print(f"✅ Human-in-the-Loop 피드백 저장 완료: Label={label}")
        return jsonify({"status": "success", "message": f"Feedback {label} saved"})
        
    except Exception as e:
        print(f"⚠️ 피드백 저장 오류: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# --- [신규 추가] 리워드 교환 엔드포인트 ---
@app.route('/redeem_reward', methods=['POST'])
def redeem_reward():
    global user_stats
    data = request.get_json()
    cost = data.get('cost', 0)

    if cost <= 0:
        return jsonify({"status": "error", "message": "유효하지 않은 비용입니다."}), 400

    try:
        # DB에서 현재 포인트를 다시 한 번 확인 (동시성 문제 방지)
        conn = sqlite3.connect('posture_data.db', check_same_thread=False)
        cursor = conn.cursor()
        
        # tts_lock을 DB 접근에도 사용하여 동시 접근 방지 (안정성 강화)
        with tts_lock: 
            cursor.execute('SELECT experience_points FROM user_gamification WHERE id = 1')
            result = cursor.fetchone()
            
            if not result:
                conn.close()
                return jsonify({"status": "error", "message": "사용자 정보를 찾을 수 없습니다."}), 404

            current_points = result[0]

            if current_points < cost:
                conn.close()
                return jsonify({"status": "error", "message": "포인트가 부족합니다."}), 400

            # 포인트 차감
            new_points = current_points - cost
            cursor.execute('UPDATE user_gamification SET experience_points = ?, updated_at = CURRENT_TIMESTAMP WHERE id = 1', (new_points,))
            conn.commit()
            
            # 전역 변수(user_stats)에도 즉시 반영
            user_stats['experience'] = new_points
            print(f"💰 리워드 교환: {cost} WP 사용. 남은 WP: {new_points}")

        conn.close()
        return jsonify({"status": "success", "message": "교환 완료", "new_points": new_points})

    except Exception as e:
        print(f"⚠️ 리워드 교환 오류: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
# --- [신규 추가 완료] ---

# [교체] /get_posture_types API 함수
@app.route('/get_posture_types', methods=['GET'])
def get_posture_types():
    try:
        # 1. dashboard.py에서 보낸 날짜 값을 받음
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')

        # 2. 날짜 값이 없으면 '최근 7일'을 기본값으로 사용
        if not start_date or not end_date:
            today = datetime.date.today()
            start_date = (today - datetime.timedelta(days=6)).isoformat()
            end_date = today.isoformat()

        conn = sqlite3.connect('posture_data.db', check_same_thread=False); cursor = conn.cursor()
        
        # 3. [수정] SQL 쿼리가 날짜 범위를 사용하도록 변경
        query = """
            SELECT posture_type, COUNT(*) as count 
            FROM posture_types_log 
            WHERE date(timestamp) BETWEEN ? AND ?
            GROUP BY posture_type 
            ORDER BY count DESC
        """
        cursor.execute(query, (start_date, end_date))
        
        results = cursor.fetchall()
        posture_stats = {row[0]: row[1] for row in results}
        conn.close()
        return jsonify({"status": "success", "posture_stats": posture_stats})
        
    except Exception as e: 
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/create_smart_notifications', methods=['POST'])
def create_smart_notifications_endpoint():
    try:
        create_smart_notifications()
        return jsonify({"status": "success", "message": "Smart notifications created"})
    except Exception as e: return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/get_smart_notifications', methods=['GET'])
def get_smart_notifications():
    try:
        conn = sqlite3.connect('posture_data.db', check_same_thread=False); cursor = conn.cursor()
        cursor.execute('''SELECT notification_type, trigger_time, last_triggered 
                          FROM smart_notifications WHERE is_active = 1 ORDER BY trigger_time''')
        notifications = cursor.fetchall()
        conn.close()
        notification_list = [{"type": notif[0], "trigger_time": notif[1], "last_triggered": notif[2]} for notif in notifications]
        return jsonify({"status": "success", "notifications": notification_list})
    except Exception as e: return jsonify({"status": "error", "message": str(e)}), 500

# [교체] /analyze_patterns API 함수
@app.route('/analyze_patterns', methods=['GET'])
def analyze_patterns():
    try:
        # 1. dashboard.py에서 보낸 날짜 값을 받음
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')

        # 2. 날짜 값이 없으면 (예: tab7) '최근 7일'을 기본값으로 사용
        if not start_date or not end_date:
            today = datetime.date.today()
            start_date = (today - datetime.timedelta(days=6)).isoformat()
            end_date = today.isoformat()
            
        # 3. [수정] 헬퍼 함수에 날짜 인자 전달
        worst_hours = analyze_posture_patterns(start_date, end_date)
        
        return jsonify({"status": "success", "worst_hours": worst_hours})
        
    except Exception as e: 
        return jsonify({"status": "error", "message": str(e)}), 500



@app.route('/set_desktop_alert', methods=['POST'])
def set_desktop_alert():
    data = request.get_json()
    enabled = data.get('enabled', True)
    with lock:
        app_state['desktop_alert_on'] = enabled
    print(f"🖥️ Desktop Alert status changed to: {'ON' if enabled else 'OFF'}")
    return jsonify({"status": "success", "desktop_alert_on": enabled})

# --- 메인 실행 로직 ---
def main():
    init_db()
    load_model()
    load_stats_from_db()
    load_user_gamification() # 게임화 데이터 로드
    # init_tts() # TTS 엔진 초기화는 worker가 담당

    stats_thread = threading.Thread(target=track_posture_stats, daemon=True)
    stats_thread.start()
    
    tts_thread = threading.Thread(target=tts_worker, daemon=True)
    tts_thread.start()

    print("🚀 AI 자세 교정 코치 서버가 시작되었습니다!")
    app.run(host='0.0.0.0', port=5000, threaded=True)

if __name__ == '__main__':
    main()