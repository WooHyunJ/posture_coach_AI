import streamlit as st
import pandas as pd
import sqlite3
import requests
import time
import plotly.express as px # Plotly Express import
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os
import datetime
from datetime import date # [신규 추가]
from dateutil.relativedelta import relativedelta # [신규 추가]
import numpy as np # Confusion Matrix 라벨링 위해 추가
from imblearn.over_sampling import SMOTE # [신규 추가]
from collections import Counter # [신규 추가]

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from imblearn.over_sampling import SMOTE # [신규 추가]
from collections import Counter # [신규 추가]


# --- 페이지 설정 ---
st.set_page_config(page_title="AI 자세 교정 코치", page_icon="🧘", layout="wide")

FLASK_SERVER_URL = "http://127.0.0.1:5000"

# --- 함수 정의 ---
def get_date_range(option):
    """선택한 옵션에 따라 시작일과 종료일을 반환합니다."""
    today = date.today()
    
    if option == "오늘":
        start_date = today
        end_date = today
    elif option == "최근 7일":
        start_date = today - datetime.timedelta(days=6)
        end_date = today
    elif option == "최근 30일":
        start_date = today - datetime.timedelta(days=29)
        end_date = today
    elif option == "이번 달":
        start_date = today.replace(day=1)
        end_date = today
    elif option == "지난 달":
        last_month_end = today.replace(day=1) - datetime.timedelta(days=1)
        start_date = last_month_end.replace(day=1)
        end_date = last_month_end
    elif option == "전체 기간":
        start_date = date(2020, 1, 1) # (아주 먼 과거)
        end_date = today
    
    # .isoformat()으로 'YYYY-MM-DD' 형식의 문자열로 반환
    return start_date.isoformat(), end_date.isoformat()

# --- 새로운 기능: 자세 교정 운동 가이드 ---
def get_exercise_guide():
    """자세 교정을 위한 운동 가이드 데이터"""
    exercises = {
        "목 스트레칭": {
            "description": "거북목 개선을 위한 목 스트레칭",
            "steps": [
                "1. 천천히 목을 좌측으로 기울여 10초간 유지",
                "2. 천천히 목을 우측으로 기울여 10초간 유지", 
                "3. 목을 앞으로 굽혀 10초간 유지",
                "4. 목을 뒤로 젖혀 10초간 유지",
                "5. 목을 좌우로 천천히 돌리기 (각 방향 5회)"
            ],
            "duration": "5분",
            "benefits": "목 근육 긴장 완화, 거북목 개선"
        },
        "어깨 스트레칭": {
            "description": "어깨와 상체 자세 개선",
            "steps": [
                "1. 양팔을 위로 뻗어 10초간 유지",
                "2. 양팔을 뒤로 젖혀 가슴 펴기 (10초)",
                "3. 어깨를 위아래로 움직이기 (10회)",
                "4. 어깨를 앞뒤로 돌리기 (각 방향 10회)",
                "5. 팔을 가로로 뻗어 반대편 어깨 당기기"
            ],
            "duration": "7분",
            "benefits": "어깨 근육 이완, 상체 자세 개선"
        },
        "등 스트레칭": {
            "description": "등과 척추 자세 교정",
            "steps": [
                "1. 의자에 앉아 허리를 곧게 펴기",
                "2. 양팔을 뒤로 젖혀 가슴 펴기 (15초)",
                "3. 허리를 좌우로 천천히 돌리기 (각 방향 5회)",
                "4. 상체를 앞으로 굽혀 등 스트레칭 (15초)",
                "5. 의자에서 일어나서 전신 스트레칭"
            ],
            "duration": "10분",
            "benefits": "척추 정렬 개선, 허리 통증 완화"
        },
        "깊은 호흡 운동": {
            "description": "스트레스 완화 및 자세 개선",
            "steps": [
                "1. 편안한 자세로 앉기",
                "2. 코로 깊게 숨을 들이마시기 (4초)",
                "3. 숨을 참고 있기 (4초)",
                "4. 입으로 천천히 숨 내쉬기 (6초)",
                "5. 5-10회 반복"
            ],
            "duration": "3분",
            "benefits": "스트레스 완화, 자세 개선, 집중력 향상"
        }
    }
    return exercises

# --- 새로운 기능: 목표 설정 및 진행률 추적 ---
def init_goals_db():
    """목표 설정을 위한 데이터베이스 초기화"""
    conn = sqlite3.connect('posture_data.db', check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS goals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        goal_type TEXT NOT NULL,
        target_value REAL NOT NULL,
        current_value REAL DEFAULT 0,
        start_date TEXT NOT NULL,
        end_date TEXT NOT NULL,
        is_active BOOLEAN DEFAULT 1,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.commit()
    conn.close()

def save_goal(goal_type, target_value, start_date, end_date):
    """새로운 목표 저장"""
    conn = sqlite3.connect('posture_data.db', check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''INSERT INTO goals (goal_type, target_value, start_date, end_date, is_active) 
                      VALUES (?, ?, ?, ?, 1)''', (goal_type, target_value, start_date, end_date))
    conn.commit()
    conn.close()

def get_active_goals(is_active_status=True):
    """활성화 또는 비활성화된 목표들 가져오기"""
    conn = sqlite3.connect('posture_data.db', check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM goals WHERE is_active = ? ORDER BY created_at DESC', (is_active_status,))
    goals = cursor.fetchall()
    conn.close()
    return goals

def update_goal_progress(goal_id, current_value):
    """목표 진행률 업데이트"""
    conn = sqlite3.connect('posture_data.db', check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('UPDATE goals SET current_value = ? WHERE id = ?', (current_value, goal_id))
    conn.commit()
    conn.close()

def calculate_goal_progress(goal_type, start_date, end_date):
    """목표 진행률 계산"""
    conn = sqlite3.connect('posture_data.db', check_same_thread=False)
    cursor = conn.cursor()
    
    if goal_type == "daily_good_posture_minutes":
        cursor.execute('''SELECT COALESCE(SUM(good_seconds), 0) / 60.0 
                          FROM daily_stats 
                          WHERE date = ?''', (datetime.date.today().isoformat(),))
        result = cursor.fetchone()
        return result[0] if result else 0
    
    elif goal_type == "weekly_good_posture_hours":
        start_week = datetime.date.today() - datetime.timedelta(days=7)
        cursor.execute('''SELECT COALESCE(SUM(good_seconds), 0) / 3600.0 
                          FROM daily_stats 
                          WHERE date >= ? AND date <= ?''', 
                       (start_week.isoformat(), datetime.date.today().isoformat()))
        result = cursor.fetchone()
        return result[0] if result else 0
    
    elif goal_type == "monthly_bad_posture_reduction":
        start_month = datetime.date.today() - datetime.timedelta(days=30)
        cursor.execute('''SELECT COALESCE(SUM(good_seconds), 0) as good, 
                                 COALESCE(SUM(bad_seconds), 0) as bad
                          FROM daily_stats 
                          WHERE date >= ? AND date <= ?''', 
                       (start_month.isoformat(), datetime.date.today().isoformat()))
        result = cursor.fetchone()
        if result and (result[0] + result[1]) > 0:
            good_ratio = result[0] / (result[0] + result[1])
            return good_ratio * 100  # 백분율로 반환
        return 0
    
    conn.close()
    return 0

# --- 새로운 기능: 음성 피드백 ---
def send_voice_feedback(message):
    """서버에 음성 피드백 요청 전송"""
    try:
        response = requests.post(f"{FLASK_SERVER_URL}/voice_feedback", 
                                 json={"message": message})
        return response.status_code == 200
    except:
        return False

# --- [교체] 기존 clear_database() 함수를 이걸로 교체 ---

def clear_database():
    """
    [수정됨] 
    서버의 /clear_data 엔드포인트를 호출하고, 성공/실패 여부(boolean)만 반환합니다.
    UI(st.success, st.error)는 여기서 처리하고, 화면 새로고침(st.rerun)은 버튼 로직에서 직접 처리합니다.
    """
    try:
        response = requests.post(f"{FLASK_SERVER_URL}/clear_data")
        
        if response.status_code == 200:
            st.success("✅ 모든 데이터가 성공적으로 초기화되었습니다.")
            return True # 성공
        else:
            st.error(f"⚠️ 데이터베이스 초기화 실패: {response.json().get('message', '서버 오류')}")
            return False # 실패
            
    except requests.exceptions.ConnectionError:
        st.error("⚠️ 서버 연결 실패. streaming_server.py가 실행 중인지 확인하세요.")
        return False # 실패
    except Exception as e:
        st.error(f"⚠️ 데이터베이스 초기화 중 알 수 없는 오류 발생: {e}")
        return False # 실패
    
# [교체] load_data 함수 전체
def load_data(start_date=None, end_date=None):
    """
    posture_log 테이블에서 데이터를 로드합니다.
    날짜 범위가 주어지면 해당 기간의 데이터만 필터링합니다.
    """
    try:
        conn = sqlite3.connect('posture_data.db', check_same_thread=False)
        
        # 날짜 범위가 없으면 (예: tab1에서 학습 시) 모든 데이터를 로드
        if start_date is None or end_date is None:
            query = "SELECT * from posture_log WHERE label IS NOT NULL"
            params = ()
        # 날짜 범위가 있으면 (예: tab2에서 리포트 시) 데이터 필터링
        else:
            # [수정] date() 함수를 사용해 timestamp의 날짜 부분만 비교
            query = "SELECT * from posture_log WHERE label IS NOT NULL AND date(timestamp) BETWEEN ? AND ?"
            params = (start_date, end_date)
            
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df
        
    except Exception as e:
        st.error(f"⚠️ 데이터베이스 로딩 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()

# [교체] train_and_save_model 함수 전체 (이상탐지, 오버샘플링 적용)
def train_and_save_model():
    df = load_data() # 1. 모든 데이터 로드 (피드백 포함)
    
    if len(df) < 50: # 데이터가 너무 적으면 중지
        st.warning("⚠️ 모델 학습을 위해 최소 50개 이상의 데이터가 필요합니다."); return
    if len(df['label'].unique()) < 2:
        st.warning("⚠️ 좋은 자세(0)와 나쁜 자세(1) 데이터가 모두 필요합니다."); return

    with st.spinner("⏳ AI 모델 학습을 시작합니다... (1/5)"):
        # --- 2. 특성 공학 ---
        df['shoulder_ear_diff_x'] = df['right_shoulder_x'] - df['right_ear_x']
        features = ['right_ear_x', 'right_ear_y', 'right_shoulder_x', 'right_shoulder_y', 'shoulder_ear_diff_x']
        X = df[features]; y = df['label']
        
        st.info(f"✅ (1/5) 총 {len(df)}개 데이터 로드 완료.")

    with st.spinner("⏳ (2/5) 악의적 데이터(이상치) 제거 중..."):
        # --- 3. [신규] 이상탐지 (IsolationForest) 적용 ---
        # "contamination=0.05" -> 전체 데이터 중 5%를 이상치로 간주하여 제거
        iso = IsolationForest(contamination=0.05, random_state=42)
        y_iso = iso.fit_predict(X)
        
        # -1이 이상치, 1이 정상 데이터
        mask = (y_iso == 1) 
        X_clean = X[mask]
        y_clean = y[mask]
        
        removed_count = len(X) - len(X_clean)
        st.info(f"✅ (2/5) 이상치 {removed_count}개 제거 완료! (총 {len(X_clean)}개 데이터 사용)")

    with st.spinner("⏳ (3/5) 학습/테스트 데이터 분리 중..."):
        # --- 4. 훈련/테스트 데이터 분리 ---
        # (주의!) 오버샘플링 전에 반드시 데이터를 분리해야 함
        X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean)
        
        st.info(f"✅ (3/5) 학습 데이터: {len(X_train)}개, 테스트 데이터: {len(X_test)}개")

    with st.spinner("⏳ (4/5) 데이터 불균형 해소(Oversampling) 중..."):
        # --- 5. [신규] 오버샘플링 (SMOTE) 적용 ---
        # (주의!) 훈련 데이터(X_train, y_train)에만 적용해야 함
        
        # 불균형 데이터 개수 확인
        st.info(f"Oversampling 전 (학습 데이터): {Counter(y_train)}")
        
        # SMOTE 적용
        try:
            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            st.info(f"Oversampling 후 (학습 데이터): {Counter(y_train_res)}")
        except ValueError as e:
            # (예외처리) 한쪽 라벨 데이터가 너무 적어(예: 5개 미만) SMOTE가 실패할 경우
            st.warning(f"⚠️ 오버샘플링(SMOTE) 실패: {e}. 원본 데이터로 학습합니다.")
            X_train_res, y_train_res = X_train, y_train # 원본 사용

    with st.spinner("⏳ (5/5) AI 모델 학습 및 평가 중..."):
        # --- 6. 모델 학습 및 저장 ---
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # [수정] SMOTE로 처리된 데이터로 학습
        model.fit(X_train_res, y_train_res) 
        
        # [유지] 평가는 원본 테스트 데이터(X_test)로 수행
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        joblib.dump(model, 'posture_model.joblib')
        
        st.success(f"✅ AI 모델 학습 완료! (예측 정확도: {accuracy * 100:.2f}%)")
        st.info("💡 '모델 새로고침' 버튼을 눌러 서버에 적용하세요!")
       
# --- 데이터 시각화 함수들 (Plotly 크기 직접 지정 + config 추가) ---
def create_hourly_line_chart(df):
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp']); df['hour'] = df['timestamp'].dt.hour
        bad_posture_df = df[df['label'] == 1]
        if bad_posture_df.empty: st.info("ℹ️ 나쁜 자세 데이터 없음."); return
        hourly_counts = bad_posture_df.groupby('hour').size().reset_index(name='count')
        peak_hour, peak_count = 0, 0
        if not hourly_counts.empty:
            peak_hour = hourly_counts.loc[hourly_counts['count'].idxmax()]['hour']; peak_count = hourly_counts['count'].max()
            st.metric(label="🚨 최다 발생 시간", value=f"{int(peak_hour)}시", delta=f"{peak_count}회", delta_color="inverse")
        fig = px.line(hourly_counts, x='hour', y='count', markers=True, title='시간대별 거북목 발생 추세')
        fig.update_layout(width=600, height=300, title_font_size=16)
        st.plotly_chart(fig, config={'displayModeBar': True})
        with st.expander("📈 해석"):
             st.markdown(f"""
             이 그래프는 하루 중 **언제** 나쁜 자세(거북목)가 많이 발생하는지 추세를 보여줍니다.
             - 데이터에 따르면, **{int(peak_hour) if not hourly_counts.empty else 'N/A'}시**에 거북목 자세가 가장 빈번하게 나타났습니다.
             - 이 시간대에 의식적으로 자세를 점검하거나 휴식을 취하는 것이 좋겠습니다.
             """)
    except Exception as e: st.error(f"⚠️ 시간대별 그래프 오류: {e}")

def create_ratio_bar_chart(df):
    try:
        if df.empty: return
        posture_counts = df['label'].value_counts()
        good_count = posture_counts.get(0, 0); bad_count = posture_counts.get(1, 0); total_count = good_count + bad_count
        data = {'Status': ['좋은 자세 (Good)', '나쁜 자세 (Bad)'], 'Count': [good_count, bad_count]}
        count_df = pd.DataFrame(data)
        bad_ratio = 0
        if total_count > 0: bad_ratio = (bad_count / total_count) * 100; st.metric(label="📉 나쁜 자세 비율", value=f"{bad_ratio:.1f}%")
        fig = px.bar(count_df, x='Status', y='Count', text='Count', title='전체 자세 데이터 개수', color='Status', color_discrete_map={'좋은 자세 (Good)': '#4CAF50', '나쁜 자세 (Bad)': '#F44336'})
        fig.update_layout(width=400, height=280, title_font_size=16, xaxis_title='')
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, config={'displayModeBar': True})
        with st.expander("📊 해석"):
            st.markdown(f"""
            이 그래프는 지금까지 수집된 전체 데이터 중 **좋은 자세와 나쁜 자세가 각각 몇 개씩 있는지** 보여줍니다.
            - 전체 데이터 {total_count}개 중 약 **{bad_ratio:.1f}%** 가 나쁜 자세로 기록되었습니다.
            - 학습 데이터의 균형(좋은 자세와 나쁜 자세의 비율)을 확인하는 데 도움이 됩니다.
            """)
    except Exception as e: st.error(f"⚠️ 자세 데이터 개수 그래프 오류: {e}")

def create_scatter_plot(df):
    try:
        if df.empty: return
        df['shoulder_ear_diff_x'] = df['right_shoulder_x'] - df['right_ear_x']; df['label_str'] = df['label'].map({0: '좋은 자세 (0)', 1: '나쁜 자세 (1)'})
        fig = px.scatter(df, x='right_shoulder_x', y='right_ear_x', color='label_str', title='AI의 자세 판단 기준 시각화', labels={'right_shoulder_x': '어깨 X', 'right_ear_x': '귀 X', 'label_str': 'Label'}, color_discrete_map={'좋은 자세 (0)': '#4CAF50', '나쁜 자세 (1)': '#F44336'})
        fig.update_layout(width=500, height=350, title_font_size=16)
        st.plotly_chart(fig, config={'displayModeBar': True})
        with st.expander("🤖 해석"):
            st.markdown("""
            이 그래프는 AI가 **어떤 특징(어깨와 귀의 상대적 위치)을 기준으로** 좋은 자세(초록)와 나쁜 자세(빨강)를 구분하는지 시각적으로 보여줍니다.
            - X축은 어깨의 좌우 위치, Y축은 귀의 좌우 위치를 나타냅니다 (웹캠 기준).
            - 두 그룹의 점들이 잘 분리될수록 AI가 명확하게 학습되었음을 의미합니다.
            """)
    except Exception as e: st.error(f"⚠️ AI 판단 기준 그래프 오류: {e}")




def create_confusion_matrix_plot(df):
    try:
        if not os.path.exists('posture_model.joblib'): st.warning("⚠️ 모델 파일 없음."); return
        if df.empty: st.warning("⚠️ 평가 데이터 없음."); return
        model = joblib.load('posture_model.joblib'); df['shoulder_ear_diff_x'] = df['right_shoulder_x'] - df['right_ear_x']
        features = ['right_ear_x', 'right_ear_y', 'right_shoulder_x', 'right_shoulder_y', 'shoulder_ear_diff_x']
        X = df[features]; y_true = df['label']; y_pred = model.predict(X)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1]); cm_text = [[str(y) for y in x] for x in cm]
        try:
            report = classification_report(y_true, y_pred, target_names=['좋은 자세 (0)', '나쁜 자세 (1)'], output_dict=True, zero_division=0)
            accuracy = report['accuracy']; precision_bad = report.get('나쁜 자세 (1)', {}).get('precision', 0); recall_bad = report.get('나쁜 자세 (1)', {}).get('recall', 0)
            col1, col2, col3 = st.columns(3); col1.metric(label="🎯 정확도", value=f"{accuracy*100:.1f}%"); col2.metric(label="🔍 정밀도(Bad)", value=f"{precision_bad*100:.1f}%"); col3.metric(label="👀 재현율(Bad)", value=f"{recall_bad*100:.1f}%")
        except: accuracy, precision_bad, recall_bad = 0, 0, 0
        fig = px.imshow(cm, text_auto=True, aspect="auto", labels=dict(x="AI 예측", y="실제 정답", color="횟수"), x=['좋음', '나쁨'], y=['좋음', '나쁨'], color_continuous_scale=px.colors.sequential.Blues, title='AI 모델 성능 (혼동 행렬)')
        fig.update_layout(width=400, height=280, title_font_size=16)
        st.plotly_chart(fig, config={'displayModeBar': True})
        with st.expander("📉 해석"):
            st.markdown(f"""
            **혼동 행렬**은 현재 학습된 AI 모델의 **'성적표'**입니다.
            - **(좌상단)** 좋은 자세를 좋다고 맞힘 (TN: {cm[0,0]}회)
            - **(우하단)** 나쁜 자세를 나쁘다고 맞힘 (TP: {cm[1,1]}회)
            - **(우상단)** 좋은 자세를 나쁘다고 틀림 (FP: {cm[0,1]}회) - AI가 너무 예민함
            - **(좌하단)** 나쁜 자세를 좋다고 틀림 (FN: {cm[1,0]}회) - AI가 너무 둔감함

            **상세 지표:**
            - **전체 정확도 (Accuracy):** 모델이 전체 데이터 중 얼마나 정확하게 예측했는지 ({accuracy*100:.1f}%).
            - **나쁜자세 정밀도 (Precision):** AI가 '나쁨'이라고 예측했을 때, 실제로 나쁜 자세일 확률 ({precision_bad*100:.1f}%).
            - **나쁜자세 재현율 (Recall):** 실제 나쁜 자세 중에서 AI가 얼마나 놓치지 않고 '나쁨'이라고 찾아냈는지 ({recall_bad*100:.1f}%).
            """)
    except Exception as e: st.error(f"⚠️ 혼동 행렬 그래프 오류: {e}")

# [교체] create_posture_types_chart 함수 전체
def create_posture_types_chart(start_date, end_date):
    """자세 유형별 분석 차트 생성"""
    try:
        # [수정] API 호출 시 params로 날짜 전달
        response = requests.get(f"{FLASK_SERVER_URL}/get_posture_types", 
                                params={"start_date": start_date, "end_date": end_date})
        
        if response.status_code == 200:
            posture_stats = response.json()['posture_stats']
            
            if posture_stats:
                posture_names = {
                    'good_posture': '좋은 자세', 'forward_head': '거북목',
                    'rounded_shoulders': '구부정한 어깨', 'slouching': '구부정한 등',
                    'shoulder_tilt': '어깨 기울어짐', 'head_tilt': '머리 기울어짐'
                }
                chart_data = []
                for posture_type, count in posture_stats.items():
                    chart_data.append({'자세 유형': posture_names.get(posture_type, posture_type), '횟수': count})
                
                if chart_data:
                    df_chart = pd.DataFrame(chart_data)
                    fig_pie = px.pie(df_chart, values='횟수', names='자세 유형', 
                                     title='자세 유형별 분포 (선택 기간)',
                                     color_discrete_sequence=px.colors.qualitative.Set3)
                    fig_pie.update_layout(width=500, height=400)
                    st.plotly_chart(fig_pie, config={'displayModeBar': True})
                    
                    fig_bar = px.bar(df_chart, x='자세 유형', y='횟수', 
                                     title='자세 유형별 발생 횟수 (선택 기간)', color='횟수',
                                     color_continuous_scale='RdYlBu_r')
                    fig_bar.update_layout(width=600, height=400)
                    st.plotly_chart(fig_bar, config={'displayModeBar': True})
                    
                    with st.expander("📊 해석"):
                        total_count = sum(posture_stats.values())
                        good_ratio = posture_stats.get('good_posture', 0) / total_count * 100 if total_count > 0 else 0
                        bad_postures = {k: v for k, v in posture_stats.items() if k != 'good_posture'}
                        worst_posture = "N/A"
                        if bad_postures:
                            worst_posture_type = max(bad_postures.items(), key=lambda x: x[1])[0]
                            worst_posture = posture_names.get(worst_posture_type, worst_posture_type)

                        st.markdown(f"""
                        이 그래프는 선택한 기간 동안 감지된 **다양한 자세 유형**의 분포를 보여줍니다.
                        - 가장 많이 발생하는 자세 문제: **{worst_posture}**
                        - 좋은 자세 비율: **{good_ratio:.1f}%**
                        - 개선이 필요한 자세 유형을 파악하여 맞춤형 운동 계획을 세울 수 있습니다.
                        """)
                else:
                    st.info("ℹ️ 해당 기간에 자세 유형 데이터가 없습니다. (chart_data 비어있음)")
            else:
                st.info("ℹ️ 해당 기간에 자세 유형 데이터가 없습니다. 실시간 코칭을 사용해보세요!")
        else:
            st.error("자세 유형 데이터를 불러올 수 없습니다. (서버 응답)")
    except Exception as e:
        st.error(f"자세 유형 분석 오류: {e}")

# [교체] create_pattern_analysis_chart 함수 전체
def create_pattern_analysis_chart(start_date, end_date):
    """패턴 분석 리포트 차트 생성"""
    try:
        # [수정] API 호출 시 params로 날짜 전달
        response = requests.get(f"{FLASK_SERVER_URL}/analyze_patterns", 
                                params={"start_date": start_date, "end_date": end_date})
        
        if response.status_code == 200:
            worst_hours = response.json()['worst_hours']
            
            if worst_hours:
                hours_data = []
                for hour, freq in worst_hours:
                    hours_data.append({'시간대': f"{hour}시", '나쁜 자세 횟수': freq})
                
                df_hours = pd.DataFrame(hours_data)
                fig = px.bar(df_hours, x='시간대', y='나쁜 자세 횟수', 
                             title='시간대별 나쁜 자세 발생 빈도 (선택 기간)',
                             color='나쁜 자세 횟수',
                             color_continuous_scale='Reds')
                fig.update_layout(width=600, height=400)
                st.plotly_chart(fig, config={'displayModeBar': True})
                
                st.subheader("🔍 패턴 분석 결과")
                col1, col2, col3 = st.columns(3)
                
                worst_hour = 0
                total_bad = sum(freq for _, freq in worst_hours)
                avg_bad = total_bad / len(worst_hours) if worst_hours else 0
                
                if worst_hours:
                    worst_hour = worst_hours[0][0]
                
                with col1:
                    st.metric("🚨 최악 시간대", f"{worst_hour}시", 
                              delta=f"{worst_hours[0][1]}회" if worst_hours else "0회")
                with col2:
                    st.metric("📉 총 나쁜 자세", f"{total_bad}회")
                with col3:
                    st.metric("📊 평균 발생", f"{avg_bad:.1f}회/시간")
                
                with st.expander("💡 패턴 분석 해석"):
                    st.markdown(f"""
                    **패턴 분석 결과 (선택 기간):**
                    1. **가장 문제가 많은 시간대**: {worst_hour}시
                       - 이 시간대에 특히 주의가 필요합니다.
                    2. **총 나쁜 자세 발생**: {total_bad}회 (선택 기간)
                    3. **개선 권장사항**:
                       - {worst_hour}시 1시간 전에 알림을 설정하여 자세를 점검하세요.
                       - 해당 시간대에는 더 자주 스트레칭을 하세요.
                    """)
            else:
                st.info("ℹ️ 해당 기간에 충분한 데이터가 없습니다.")
        else:
            st.error("패턴 분석에 실패했습니다. (서버 응답)")
    except Exception as e:
        st.error(f"패턴 분석 오류: {e}")


def update_goal_status(goal_id, is_active=False):
    """목표의 활성 상태를 변경 (완료 처리)"""
    conn = sqlite3.connect('posture_data.db', check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('UPDATE goals SET is_active = ? WHERE id = ?', (is_active, goal_id))
    conn.commit()
    conn.close()

def delete_goal(goal_id):
    """목표를 DB에서 영구 삭제"""
    conn = sqlite3.connect('posture_data.db', check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM goals WHERE id = ?', (goal_id,))
    conn.commit()
    conn.close()

def get_goal_recommendation():
    """최근 7일간의 데이터를 바탕으로 목표 값을 추천"""
    conn = sqlite3.connect('posture_data.db', check_same_thread=False)
    cursor = conn.cursor()
    start_week = (datetime.date.today() - datetime.timedelta(days=7)).isoformat()
    
    # 최근 7일간의 '일일 좋은 자세 시간(분)' 평균 계산
    cursor.execute('''SELECT COALESCE(AVG(good_seconds), 0) / 60.0 
                      FROM daily_stats 
                      WHERE date >= ?''', (start_week,))
    avg_minutes = cursor.fetchone()[0]
    conn.close()
    
    if avg_minutes < 10:
        return 15.0 # 최소 목표
    else:
        # 기존 평균보다 약 20% 높은 값을 5분 단위로 올림
        recommended_value = (avg_minutes * 1.2)
        return (recommended_value // 5 + 1) * 5
    
# ==================================================================
# --- Streamlit 앱 메인 UI ---
# ==================================================================
st.title("🤖 AI 자세 교정 코치")
st.caption("개인 맞춤형 AI를 통해 실시간으로 자세를 교정하고, 데이터 분석을 통해 자세 습관을 개선하세요.")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "⚙️ **AI 모델 설정**", 
    "📊 **자세 분석 리포트**", 
    "👨‍🏫 **실시간 AI 코칭**", 
    "🏃‍♂️ **자세 교정 운동**", 
    "🎯 **목표 설정 & 진행률**",
    "🎮 **웰니스 챌린지**",
    "🔔 **스마트 알림 관리**"
])

# --- AI 모델 설정 탭 ---
# --- [교체] 기존 tab1 블록 전체를 아래 코드로 교체 ---




# --- AI 모델 설정 탭 ---
with tab1:
    st.header("⚙️ AI 모델 개인 맞춤 설정")
    st.markdown("""
    이 탭에서는 **개인 맞춤형 AI 모델**을 생성하고 관리합니다. **처음 사용 시** 아래 순서대로 진행하세요.
    1.  **(선택)** '데이터 초기화' 버튼으로 이전 기록을 삭제합니다.
    2.  '웹캠 켜기' 버튼 후, 안내에 따라 '좋은 자세'와 '나쁜 자세' 데이터를 각각 30초씩 수집합니다.
    3.  'AI 모델 학습 실행' 버튼으로 모델을 학습시킵니다.
    4.  **'모델 새로고침' 버튼을 반드시 눌러** 서버에 새 모델을 적용합니다.
    """)
    
    # --- [핵심 수정] 2단계 삭제 확인 로직 (버튼 크기, 화면 복귀) ---
    with st.expander("⚠️ **데이터 초기화 (주의!)**"):
        st.warning("이 버튼을 누르면 기존에 저장된 모든 자세 데이터와 통계가 영구적으로 삭제됩니다.")
        
        if 'confirm_delete' not in st.session_state:
            st.session_state.confirm_delete = False

        # 확인 상태가 아닐 때 -> '초기화' 버튼 표시
        if not st.session_state.confirm_delete:
            # [수정] use_container_width=True 제거
            if st.button("모든 데이터 초기화 및 재시작"): 
                st.session_state.confirm_delete = True
                st.rerun() 

        # 확인 상태일 때 -> '예/아니요' 버튼 표시
        if st.session_state.confirm_delete:
            st.error("#### 🚨 정말 초기화 하시겠습니까? 🚨\n모든 WP, 통계, 목표 기록이 영구적으로 삭제됩니다.")
            
            col1, col2, col3 = st.columns([0.4, 0.4, 0.2]) # 버튼 정렬을 위해 3칸으로
            with col1:
                # [수정] use_container_width=True 제거
                if st.button("🔴 예, 영구적으로 삭제합니다"):
                    clear_database() # API 호출 (성공/실패 메시지 표시)
                    st.session_state.confirm_delete = False # 상태 되돌리기
                    st.rerun() # "처음 화면"으로 돌아가기
            with col2:
                # [수정] use_container_width=True 제거
                if st.button("🟢 아니요, 취소합니다"):
                    st.session_state.confirm_delete = False # 상태 되돌리기
                    st.rerun() # "처음 화면"으로 돌아가기
    # --- [수정 완료] ---
    
    st.divider()

    col_control, col_video = st.columns([0.4, 0.6])

    with col_control:
        st.subheader("📹 실시간 영상 제어")
        if 'show_camera' not in st.session_state: st.session_state.show_camera = False
        if st.button("▶️ 웹캠 켜기", use_container_width=True): st.session_state.show_camera = True
        if st.button("⏹️ 웹캠 끄기", use_container_width=True): st.session_state.show_camera = False
        st.divider()

        if st.session_state.get('show_camera', False):
            st.subheader("1. 개인 맞춤 데이터 수집")
            st.caption("각 버튼을 누르면 **30초**간 데이터가 수집됩니다.")
            with st.expander("💡 올바른 데이터 수집 방법"):
                st.markdown("""
                - **좋은 자세:** 허리를 펴고 턱을 당기되, **오래 유지할 수 있는 가장 편안하면서도 이상적인 자세**를 기준으로 삼으세요. 수집 중 **약간씩 자연스럽게 움직여** AI에게 '좋은 자세의 범위'를 알려주는 것이 중요합니다.
                - **나쁜 자세:** **평소 무의식적으로 취하게 되는 불편한 자세** (예: 목을 앞으로 빼거나 어깨가 말리는 자세)를 기준으로 삼으세요. 마찬가지로 약간씩 움직이며 '나쁜 자세의 범위'를 알려주세요.
                """)
            progress_bar = st.progress(0.0, text="데이터 수집 대기 중...")
            if st.button("✅ '좋은 자세' 데이터 수집"):
                try:
                    requests.post(f"{FLASK_SERVER_URL}/set_mode", json={"mode": "collect", "label": 0, "duration": 30})
                    for i in range(30): progress_bar.progress((i+1)/30, text=f"좋은 자세 수집 중... ({i+1}/30초)"); time.sleep(1)
                    st.success("✅ 좋은 자세 데이터 수집 완료!")
                except: st.error("⚠️ 서버 연결 실패")
                finally: progress_bar.progress(0.0, text="데이터 수집 대기 중...")
            if st.button("❌ '나쁜 자세' 데이터 수집"):
                try:
                    requests.post(f"{FLASK_SERVER_URL}/set_mode", json={"mode": "collect", "label": 1, "duration": 30})
                    for i in range(30): progress_bar.progress((i+1)/30, text=f"나쁜 자세 수집 중... ({i+1}/30초)"); time.sleep(1)
                    st.success("❌ 나쁜 자세 데이터 수집 완료!")
                except: st.error("⚠️ 서버 연결 실패")
                finally: progress_bar.progress(0.0, text="데이터 수집 대기 중...")
            st.divider()
            st.subheader("2. AI 모델 학습")
            st.caption("수집된 모든 데이터를 사용하여 AI 모델을 학습시킵니다.")
            if st.button("🧠 AI 모델 학습 실행"): train_and_save_model()
            st.info("💡 학습 후에는 아래 '모델 새로고침' 버튼을 꼭 눌러 서버에 새 모델을 적용해야 합니다.")
            if st.button("🔄 모델 새로고침"):
                 with st.spinner("⏳ 서버에 새 모델을 적용하는 중..."):
                    # [핵심 수정] 파일 I/O(쓰기/읽기) 충돌을 피하기 위해 1초 대기
                     time.sleep(1) 
                     try:
                        requests.post(f"{FLASK_SERVER_URL}/reload_model")
                        st.success("✅ 서버 모델 새로고침 완료!")
                     except: 
                        st.error("⚠️ 서버 연결 실패")

            # --- [신규 추가] 자동 재학습 및 평가 버튼 ---
            st.divider()
            st.subheader("3. AI 자동 재학습 (Human-in-the-Loop)")
            st.info("실시간 코칭 중 'AI 피드백' 버튼으로 '오답 노트'를 충분히 모은 뒤, 이 버튼을 눌러 AI를 더 똑똑하게 만드세요.")
            
            if st.button("🚀 최신 피드백으로 AI 자동 재학습/평가", use_container_width=True):
                
                # 1. 모든 데이터 불러오기 (초기 데이터 + 피드백 데이터)
                with st.spinner("⏳ (1/4) 모든 학습 데이터를 불러오는 중..."):
                    df = load_data()
                    if df.empty or len(df) < 50:
                        st.error("⚠️ 데이터가 부족합니다. '좋은/나쁜 자세' 데이터를 더 수집해주세요.")
                        st.stop() # 작업 중단
                    
                    st.success(f"✅ 총 {len(df)}개의 학습 데이터를 불러왔습니다.")
        
                train_and_save_model() 
                

                # 4. 서버에 새 모델 적용
                with st.spinner("⏳ (4/4) 똑똑해진 새 모델을 서버에 적용 중..."):
                    try:
                        requests.post(f"{FLASK_SERVER_URL}/reload_model")
                        st.success("✅ 모든 작업 완료! 서버에 새 모델이 적용되었습니다.")
                        st.balloons()
                    except Exception as e:
                        st.error("⚠️ 서버에 새 모델 적용을 실패했습니다. 서버를 확인하세요.")
            # --- [신규 추가 완료] ---
        else:
            st.info("ℹ️ 데이터 수집 및 모델 학습을 위해 먼저 웹캠을 켜주세요.")

    with col_video:
        st.subheader("🖥️ 실시간 영상")
        if st.session_state.get('show_camera', False):
            st.image(f"{FLASK_SERVER_URL}/video_feed", use_column_width=True)
        else:
            st.info("ℹ️ 웹캠을 시작하려면 '▶️ 웹캠 켜기' 버튼을 눌러주세요.")
            

with tab2:
    st.header("📊 자세 분석 리포트")
    
    # --- [신규 추가] 날짜 필터 UI ---
    st.info("아래에서 기간을 선택하여, 해당 기간의 자세 습관과 AI 모델의 성능을 확인하세요.")
    
    col_filter1, col_filter2 = st.columns([0.4, 0.6])
    with col_filter1:
        date_option = st.selectbox(
            "📅 기간 선택:",
            options=["오늘", "최근 7일", "최근 30일", "이번 달", "지난 달", "전체 기간"],
            index=2 # 기본값 '최근 30일'
        )
    
    # 선택한 옵션에 따라 날짜 계산
    start_date, end_date = get_date_range(date_option)
    
    with col_filter2:
        st.caption(f"&nbsp; \n &nbsp; \n 선택된 기간: **{start_date}** ~ **{end_date}**")
    # --- [신규 추가 완료] ---
    
    chart_options = ['자세 유형별 분석', '패턴 분석 리포트', '시간대별 추세', '자세 데이터 개수', 'AI 판단 기준', '모델 성능 분석']
    selected_chart = st.selectbox("분석 차트 선택:", options=chart_options, label_visibility="collapsed")
            
    st.divider()
    
    # [수정] 필터링된 날짜로 학습 데이터를 로드합니다 (차트용)
    df_filtered = load_data(start_date, end_date) 

    
    if selected_chart == '자세 유형별 분석':
        # [수정] 날짜 인자 전달
        create_posture_types_chart(start_date, end_date)
        
    elif selected_chart == '패턴 분석 리포트':
        # [수정] 날짜 인자 전달
        create_pattern_analysis_chart(start_date, end_date)
        
    elif selected_chart == '시간대별 추세':
        # [수정] 필터링된 df 전달 (함수 자체는 수정 불필요)
        create_hourly_line_chart(df_filtered)
        
    elif selected_chart == '자세 데이터 개수':
        # [수정] 필터링된 df 전달 (함수 자체는 수정 불필요)
        create_ratio_bar_chart(df_filtered)
        
    elif selected_chart == 'AI 판단 기준':
        # [수정] 필터링된 df 전달 (함수 자체는 수정 불필요)
        create_scatter_plot(df_filtered)
        
    elif selected_chart == '모델 성능 분석':
        # [수정] 필터링된 df 전달 (함수 자체는 수정 불필요)
        # (주의: 이 차트는 '모델'의 성능이며, '기간'과는 무관할 수 있음)
        # (하지만 기간별 데이터로 모델 성능을 보는 것도 의미 있으므로 df_filtered 사용)
        create_confusion_matrix_plot(df_filtered)

# --- 실시간 AI 코칭 탭 ---
with tab3:
    st.header("👨‍🏫 실시간 AI 코칭")
    st.info("현재 서버에 로드된 **개인 맞춤형 AI 모델**이 실시간 웹캠 영상을 분석하여 자세를 판단하고 피드백(색상, 소리)을 제공합니다.")
    
    col_control_coach, col_video_coach = st.columns([0.4, 0.6])

    with col_control_coach:
        st.subheader("⚙️ 코칭 설정")
        if 'coaching_on' not in st.session_state: st.session_state.coaching_on = False
        btn_cols = st.columns(2)
        with btn_cols[0]:
            if st.button("▶️ 코칭 시작", use_container_width=True):
                try: requests.post(f"{FLASK_SERVER_URL}/set_mode", json={"mode": "coach"}); st.session_state.coaching_on = True
                except: st.error("⚠️ 서버 연결 실패")
        with btn_cols[1]:
            if st.button("⏹️ 코칭 중지", use_container_width=True):
                try: requests.post(f"{FLASK_SERVER_URL}/set_mode", json={"mode": "idle"}); st.session_state.coaching_on = False
                except: st.error("⚠️ 서버 연결 실패")
        st.divider()
        
        setting_cols = st.columns(2)
        with setting_cols[0]:
            if 'sound_on' not in st.session_state: st.session_state.sound_on = True
            sound_toggle = st.toggle('🔊 소리 알림', value=st.session_state.sound_on, help="나쁜 자세 감지 시 '띵동' 소리 알림을 켭니다.")
            if sound_toggle != st.session_state.sound_on:
                try: requests.post(f"{FLASK_SERVER_URL}/set_sound", json={"sound_on": sound_toggle}); st.session_state.sound_on = sound_toggle; st.toast(f"소리 알림 {'ON' if sound_toggle else 'OFF'}")
                except: st.error("⚠️ 서버 연결 실패 (소리 설정)")
        
        with setting_cols[1]:
            if 'desktop_alert_on' not in st.session_state: st.session_state.desktop_alert_on = True
            desktop_toggle = st.toggle('🖥️ 데스크탑 알림', value=st.session_state.desktop_alert_on, help="나쁜 자세 감지 시 윈도우 팝업 알림을 켭니다. (사무실 권장)")
            if desktop_toggle != st.session_state.desktop_alert_on:
                try:
                    requests.post(f"{FLASK_SERVER_URL}/set_desktop_alert", json={"enabled": desktop_toggle})
                    st.session_state.desktop_alert_on = desktop_toggle
                    st.toast(f"데스크탑 알림 {'ON' if desktop_toggle else 'OFF'}")
                except: st.error("⚠️ 서버 연결 실패 (데스크탑 알림 설정)")

        if 'cooldown' not in st.session_state: st.session_state.cooldown = 5
        cooldown_input = st.number_input('⏱️ 알림 간격 (초)', min_value=1, max_value=60, value=st.session_state.cooldown, step=1, help="나쁜 자세 알림 소리가 최소 몇 초 간격으로 울릴지 설정합니다.")
        if cooldown_input != st.session_state.cooldown:
            try: requests.post(f"{FLASK_SERVER_URL}/set_cooldown", json={"cooldown": cooldown_input}); st.session_state.cooldown = cooldown_input; st.toast(f"알림 간격 {cooldown_input}초 설정 완료")
            except: st.error("⚠️ 서버 연결 실패 (쿨타임 설정)")
            
        if 'sensitivity_level' not in st.session_state: st.session_state.sensitivity_level = 3
        sensitivity_level = st.slider("🧠 AI 민감도 조절", min_value=1, max_value=5, value=st.session_state.sensitivity_level, step=1, help="레벨이 높을수록 AI가 더 엄격하게 나쁜 자세를 감지합니다. (1: 너그러움 ~ 5: 엄격함)")
        threshold_map = {1: 0.9, 2: 0.7, 3: 0.5, 4: 0.3, 5: 0.1}
        current_threshold = threshold_map[sensitivity_level]
        if sensitivity_level != st.session_state.sensitivity_level:
            try: requests.post(f"{FLASK_SERVER_URL}/set_sensitivity", json={"threshold": current_threshold}); st.session_state.sensitivity_level = sensitivity_level; st.toast(f"AI 민감도가 레벨 {sensitivity_level} (기준값: {current_threshold:.1f})로 설정되었습니다.")
            except: st.error("⚠️ 서버 연결 실패 (민감도 설정)")
        
        st.divider()
        
        if 'voice_feedback_on' not in st.session_state: st.session_state.voice_feedback_on = True
        voice_feedback_toggle = st.toggle('🎤 음성 피드백', value=st.session_state.voice_feedback_on, help="나쁜 자세 감지 시 음성으로 자세 교정 안내를 제공합니다.")
        if voice_feedback_toggle != st.session_state.voice_feedback_on:
            try: requests.post(f"{FLASK_SERVER_URL}/set_voice_feedback", json={"enabled": voice_feedback_toggle}); st.session_state.voice_feedback_on = voice_feedback_toggle; st.toast(f"음성 피드백 {'ON' if voice_feedback_toggle else 'OFF'}")
            except: st.error("⚠️ 서버 연결 실패 (음성 피드백 설정)")
            
        # --- [신규 추가] Human-in-the-Loop 피드백 버튼 ---
        st.divider()
        st.subheader("🤖 AI 피드백 (Human-in-the-Loop)")
        st.caption("AI의 판단이 틀렸나요? 지금 바로 '정답'을 알려주세요!")
        
        col_fb1, col_fb2 = st.columns(2)
        with col_fb1:
            if st.button("👍 (AI틀림) 지금 자세 좋아요!", use_container_width=True, help="AI가 '나쁨'이라고 했지만, 실제론 '좋은' 자세일 때"):
                try:
                    # 서버에 "지금 랜드마크는 label=0 이야"라고 전송
                    requests.post(f"{FLASK_SERVER_URL}/save_feedback", json={"label": 0})
                    st.toast("✅ '좋은 자세' 피드백이 '오답 노트'에 저장되었습니다!")
                except Exception as e:
                    st.error("⚠️ 피드백 전송 실패. 서버를 확인하세요.")
        with col_fb2:
            if st.button("👎 (AI틀림) 지금 자세 나빠요!", use_container_width=True, help="AI가 '좋음'이라고 했지만, 실제론 '나쁜' 자세일 때"):
                try:
                    # 서버에 "지금 랜드마크는 label=1 이야"라고 전송
                    requests.post(f"{FLASK_SERVER_URL}/save_feedback", json={"label": 1})
                    st.toast("❌ '나쁜 자세' 피드백이 '오답 노트'에 저장되었습니다!")
                except Exception as e:
                    st.error("⚠️ 피드백 전송 실패. 서버를 확인하세요.")
        # --- [신규 추가 완료] ---


    with col_video_coach:
        st.subheader("📹 실시간 코칭 영상")
        if st.session_state.get('coaching_on', False):
            st.image(f"{FLASK_SERVER_URL}/video_feed", use_column_width=True)
        else:
            st.info("ℹ️ 코칭을 시작하려면 '▶️ 코칭 시작' 버튼을 눌러주세요.")

# --- 자세 교정 운동 탭 ---
# dashboard.py 파일에서 with tab4: 부분을 찾아 아래 코드로 교체하세요.

# --- [교체] 기존 tab4 블록 전체를 아래 코드로 교체 ---

with tab4:
    st.header("🏃‍♂️ 자세 교정 운동 가이드")
    st.info("💡 정기적인 운동으로 자세를 개선하고 건강을 유지하세요. 각 운동은 자세 교정에 특화되어 있습니다.")
    
    exercises = get_exercise_guide()
    selected_exercise = st.selectbox(
        "운동 종류를 선택하세요:",
        options=list(exercises.keys()),
        help="자세 교정에 도움이 되는 운동을 선택하세요."
    )
    
    if selected_exercise:
        exercise_data = exercises[selected_exercise]
        col1, col2 = st.columns([0.6, 0.4])
        
        with col1:
            st.subheader(f"📋 {selected_exercise}")
            st.write(f"**설명:** {exercise_data['description']}")
            st.write(f"**소요 시간:** {exercise_data['duration']}")
            st.write(f"**효과:** {exercise_data['benefits']}")
            st.subheader("📝 운동 방법")
            for step in exercise_data['steps']:
                st.write(step)
        
        with col2:
            st.subheader("🎯 운동 시작")
            st.write("운동을 시작하기 전에 준비 운동을 해주세요.")
            
            # --- [삭제] "음성 안내 시작" 버튼 로직 제거 ---
            
            # --- 타이머 상태 관리 ---
            # 세션 상태 초기화
            if 'timer_status' not in st.session_state:
                st.session_state.timer_status = 'stopped'  # 'stopped', 'running', 'paused'
            if 'timer_remaining_seconds' not in st.session_state:
                st.session_state.timer_remaining_seconds = 0
            if 'timer_total_seconds' not in st.session_state:
                st.session_state.timer_total_seconds = 0
            if 'timer_last_update' not in st.session_state:
                st.session_state.timer_last_update = 0

            # 1. 사용자 타이머 시간 설정
            try:
                default_minutes = int(exercise_data['duration'].replace('분', ''))
            except:
                default_minutes = 5 # 기본값

            timer_minutes_input = st.number_input(
                "타이머 시간 설정 (분):", 
                min_value=1, 
                max_value=60, 
                value=default_minutes, 
                step=1,
                # 타이머가 멈춘 상태일 때만 시간 변경 가능
                disabled=(st.session_state.timer_status != 'stopped') 
            )

            # 2. 타이머 제어 버튼
            btn_cols = st.columns(3)
            with btn_cols[0]:
                if st.session_state.timer_status == 'stopped':
                    if st.button("⏱️ 시작", use_container_width=True):
                        total_seconds = timer_minutes_input * 60
                        st.session_state.timer_total_seconds = total_seconds
                        st.session_state.timer_remaining_seconds = total_seconds
                        st.session_state.timer_status = 'running'
                        st.session_state.timer_last_update = time.time()
                        st.success(f"⏱️ {timer_minutes_input}분 타이머가 시작되었습니다!")
                        st.rerun()

            with btn_cols[1]:
                if st.session_state.timer_status == 'running':
                    if st.button("⏸️ 일시 중지", use_container_width=True):
                        # 현재까지 경과 시간 계산
                        elapsed_since_last = time.time() - st.session_state.timer_last_update
                        st.session_state.timer_remaining_seconds -= elapsed_since_last
                        st.session_state.timer_status = 'paused'
                        st.rerun()

            with btn_cols[0]: # 시작 버튼과 동일 위치
                if st.session_state.timer_status == 'paused':
                    if st.button("▶️ 다시 시작", use_container_width=True):
                        st.session_state.timer_status = 'running'
                        st.session_state.timer_last_update = time.time() # 시간 기준 재설정
                        st.rerun()
            
            with btn_cols[2]: # 초기화 버튼
                if st.session_state.timer_status in ['running', 'paused']:
                    if st.button("🔄 초기화", use_container_width=True):
                        st.session_state.timer_status = 'stopped'
                        st.session_state.timer_remaining_seconds = 0
                        st.rerun()

            # 3. 타이머 진행률 표시
            if st.session_state.timer_status in ['running', 'paused']:
                
                # 'running' 상태일 때만 남은 시간 실시간 차감
                if st.session_state.timer_status == 'running':
                    elapsed_since_last = time.time() - st.session_state.timer_last_update
                    st.session_state.timer_remaining_seconds -= elapsed_since_last
                    st.session_state.timer_last_update = time.time()

                remaining = max(0, st.session_state.timer_remaining_seconds)
                
                if remaining <= 0:
                    # 타이머 종료
                    st.balloons()
                    st.success("🎉 운동 완료! 수고하셨습니다!")
                    st.session_state.timer_status = 'stopped'
                    st.session_state.timer_remaining_seconds = 0
                    if st.session_state.timer_status == 'running': # 무한 루프 방지
                        st.rerun()
                else:
                    # 타이머 진행 중
                    minutes = int(remaining // 60)
                    seconds = int(remaining % 60)
                    progress = 1.0 - (remaining / max(1, st.session_state.timer_total_seconds))
                    st.progress(progress, text=f"남은 시간: {minutes:02d}:{seconds:02d}")
                    
                    # 'running' 상태일 때만 1초마다 새로고침
                    if st.session_state.timer_status == 'running':
                        time.sleep(1)
                        st.rerun()
                        

with tab5:
    st.header("🎯 목표 설정 & 진행률 추적")
    st.info("💡 개인 맞춤형 목표를 설정하고 진행률을 추적하여 자세 개선 동기를 유지하세요.")
    
    init_goals_db() # DB 테이블 확인
    
    # 1. 탭 분리 (개선안 1)
    tab5_active, tab5_completed, tab5_new = st.tabs([
        "📊 **진행 중인 목표**", 
        "🏆 **완료된 목표**", 
        "📝 **새 목표 설정**"
    ])

    # --- 새 목표 설정 탭 ---
    with tab5_new:
        st.subheader("🎯 새 목표 설정")
        
        # 2. 목표 추천 기능 (개선안 3)
        st.info("💡 **어떤 목표를 세워야 할지 모르겠나요?**")
        if st.button("📈 지난 7일 데이터로 목표 추천받기"):
            recommended_value = get_goal_recommendation()
            st.session_state.recommended_goal_value = recommended_value
            st.success(f"최근 7일간 평균 {recommended_value*0.8:.0f}분 좋은 자세를 유지하셨네요.\n**새 목표로 {recommended_value:.0f}분**은 어떠신가요? 아래 '목표 값'에 자동 입력했습니다.")
        
        # 추천 값을 세션 상태에서 가져와 number_input의 value로 사용
        default_value = st.session_state.get('recommended_goal_value', 10.0)

        col1, col2 = st.columns(2)
        with col1:
            goal_type = st.selectbox(
                "목표 유형:",
                options=["daily_good_posture_minutes", "weekly_good_posture_hours", "monthly_bad_posture_reduction"],
                format_func=lambda x: {"daily_good_posture_minutes": "일일 좋은 자세 유지 시간 (분)", "weekly_good_posture_hours": "주간 좋은 자세 유지 시간 (시간)", "monthly_bad_posture_reduction": "월간 좋은 자세 비율 (%)"}[x],
                key="goal_type_select" # 키 추가
            )
            # '일일 좋은 자세'가 선택되었을 때만 추천 값 사용
            if st.session_state.goal_type_select == "daily_good_posture_minutes":
                target_value = st.number_input("목표 값:", min_value=1.0, step=1.0, value=default_value, key="goal_target_value")
            else:
                target_value = st.number_input("목표 값:", min_value=1.0, step=1.0, value=1.0, key="goal_target_value_other") # 다른 목표는 기본값 1

        with col2:
            start_date = st.date_input("시작 날짜:", value=datetime.date.today())
            end_date = st.date_input("종료 날짜:", value=datetime.date.today() + datetime.timedelta(days=7))
        
        if st.button("🎯 이 목표 저장하기", use_container_width=True):
            if target_value > 0 and end_date > start_date:
                save_goal(goal_type, target_value, start_date.isoformat(), end_date.isoformat())
                st.success("✅ 목표가 성공적으로 저장되었습니다! '진행 중인 목표' 탭을 확인하세요.")
                st.session_state.recommended_goal_value = 10.0 # 추천 값 초기화
                st.rerun()
            else:
                st.error("⚠️ 목표 값은 0보다 커야 하고, 종료 날짜는 시작 날짜보다 늦어야 합니다.")

    # --- 진행 중인 목표 탭 ---
    with tab5_active:
        st.subheader("📊 현재 진행 중인 목표")
        active_goals = get_active_goals() # get_active_goals는 is_active=1 인 것만 가져옴
        
        if not active_goals:
            st.info("ℹ️ 현재 진행 중인 목표가 없습니다. '새 목표 설정' 탭에서 목표를 추가해보세요!")

        for goal in active_goals:
            goal_id, goal_type, target_value, _, start_date, end_date, _, _ = goal
            current_progress = calculate_goal_progress(goal_type, start_date, end_date)
            update_goal_progress(goal_id, current_progress) # DB에 현재 값 업데이트
            
            goal_type_names = {"daily_good_posture_minutes": "일일 좋은 자세 (분)", "weekly_good_posture_hours": "주간 좋은 자세 (시)", "monthly_bad_posture_reduction": "월간 좋은 자세 (%)"}
            progress_percentage = min(100.0, (current_progress / target_value) * 100.0) if target_value > 0 else 0.0
            
            with st.container(border=True):
                col1, col2 = st.columns([0.8, 0.2])
                with col1:
                    st.markdown(f"**{goal_type_names.get(goal_type, goal_type)}**")
                    st.caption(f"기간: {start_date} ~ {end_date}")
                    st.progress(progress_percentage / 100, text=f"{progress_percentage:.1f}%")
                
                with col2:
                    st.metric(label="현재/목표", value=f"{current_progress:.1f}", delta=f"{target_value:.1f}")
                
                # 3. 목표 관리 기능 (개선안 2)
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    if st.button("✔️ 완료 처리", key=f"complete_{goal_id}", use_container_width=True):
                        update_goal_status(goal_id, is_active=False)
                        st.success(f"🎉 목표를 완료 처리했습니다! '완료된 목표' 탭에서 확인할 수 있습니다.")
                        st.rerun()
                with btn_col2:
                    if st.button("🗑️ 삭제", key=f"delete_active_{goal_id}", use_container_width=True):
                        delete_goal(goal_id)
                        st.rerun()

    # --- 완료된 목표 탭 ---
    with tab5_completed:
        st.subheader("🏆 달성했거나 만료된 목표")
        
        # get_active_goals를 수정하지 않고, get_goals 함수를 새로 만들거나
        # 여기서는 get_active_goals()를 그대로 쓰되, is_active=0 인 것을 가져오도록 DB 로직 수정이 필요
        # (위에서 추가한 헬퍼 함수를 썼으므로 get_active_goals 수정)
        
        # get_active_goals 함수를 아래와 같이 수정해야 합니다. (파일 상단)
        # def get_active_goals(is_active_status=True):
        #     conn = sqlite3.connect('posture_data.db', check_same_thread=False)
        #     cursor = conn.cursor()
        #     cursor.execute('SELECT * FROM goals WHERE is_active = ? ORDER BY created_at DESC', (is_active_status,))
        #     goals = cursor.fetchall()
        #     conn.close()
        #     return goals
        
        # -----------------
        # [수정] get_active_goals 함수가 (is_active_status=True) 인자를 받도록 상단에서 수정했다고 가정
        
        # (임시) get_active_goals 함수를 수정하지 않았다면, 아래 코드로 대체합니다.
        conn_temp = sqlite3.connect('posture_data.db', check_same_thread=False)
        cursor_temp = conn_temp.cursor()
        cursor_temp.execute('SELECT * FROM goals WHERE is_active = 0 ORDER BY created_at DESC')
        completed_goals = cursor_temp.fetchall()
        conn_temp.close()
        # -----------------
        
        if not completed_goals:
            st.info("ℹ️ 아직 완료되거나 만료된 목표가 없습니다.")

        for goal in completed_goals:
            goal_id, goal_type, target_value, current_value, start_date, end_date, _, _ = goal
            goal_type_names = {"daily_good_posture_minutes": "일일 좋은 자세 (분)", "weekly_good_posture_hours": "주간 좋은 자세 (시)", "monthly_bad_posture_reduction": "월간 좋은 자세 (%)"}
            progress_percentage = min(100.0, (current_value / target_value) * 100.0) if target_value > 0 else 0.0

            with st.container(border=True):
                col1, col2, col3 = st.columns([0.6, 0.2, 0.2])
                with col1:
                    st.markdown(f"**{goal_type_names.get(goal_type, goal_type)}** (완료)")
                    st.caption(f"기간: {start_date} ~ {end_date}")
                    st.progress(progress_percentage / 100, text=f"{progress_percentage:.1f}%")
                with col2:
                    if progress_percentage >= 100:
                        st.success(f"🎉 달성!\n({current_value:.1f}/{target_value:.1f})")
                    else:
                        st.warning(f"미달성\n({current_value:.1f}/{target_value:.1f})")
                with col3:
                    if st.button(" 되돌리기", key=f"reactivate_{goal_id}", use_container_width=True, help="진행 중 목표로 되돌립니다."):
                        update_goal_status(goal_id, is_active=True)
                        st.rerun()
                    if st.button("🗑️ 삭제", key=f"delete_comp_{goal_id}", use_container_width=True):
                        delete_goal(goal_id)
                        st.rerun()


with tab6:
    st.header("🏆 웰니스 챌린지 (리워드 프로그램)")
    st.info("💡 좋은 자세를 유지하고 웰니스 포인트를 적립하세요. 적립한 포인트는 실제 보상으로 교환할 수 있습니다. (기업/단체 연동)")
    
    # 1. 기존 엔드포인트에서 데이터 가져오기
    try:
        response = requests.get(f"{FLASK_SERVER_URL}/get_user_stats")
        if response.status_code == 200:
            user_stats = response.json()['stats']
            
            # 2. '경험치(XP)'를 '웰니스 포인트(WP)'로 해석하여 표시
            total_points = user_stats.get('experience', 0)
            total_minutes = user_stats.get('total_good_minutes', 0)
            best_streak = user_stats.get('best_streak', 0)
            
            st.subheader("💰 나의 웰니스 지갑")
            cols = st.columns(3)
            cols[0].metric("🏦 총 웰니스 포인트 (WP)", f"{total_points} WP")
            cols[1].metric("📈 총 좋은 자세 시간", f"{total_minutes} 분")
            cols[2].metric("🥇 최고 연속 기록", f"{best_streak} 분")
            
            st.divider()

            # 3. '리워드 상점', '리더보드', '배지' 탭으로 UI 개편
            tab_shop, tab_leaderboard, tab_badges = st.tabs([
                "🎁 **리워드 상점**", 
                "🥇 **이달의 리더보드**", 
                "🏆 **나의 배지**"
            ])

            # --- 리워드 상점 탭 ---
            with tab_shop:
                st.subheader(f"🎁 리워드 상점 (현재: {total_points} WP)")
                st.caption("이곳의 상품은 예시이며, 실제로는 기업/단체의 복지 정책과 연동됩니다.")
                
                # 교환 가능한 상품 목록 (예시)
                rewards = {
                    "store": {"name": "🏪 편의점 5,000원권", "cost": 5000},
                    "coffee": {"name": "☕ 메가커피 5,000원권", "cost": 5000},
                    "health": {"name": "🧘 퇴근시간 30분 단축", "cost": 10000}
                }
                
                # [수정] 포인트가 가장 적게 드는 순서로 정렬 (낮은 금액부터 표시)
                sorted_rewards = sorted(rewards.items(), key=lambda item: item[1]['cost'])
                
                cols = st.columns(len(sorted_rewards))
                
                for i, (key, reward) in enumerate(sorted_rewards):
                    with cols[i]:
                        with st.container(border=True, height=200):
                            st.markdown(f"#### {reward['name']}")
                            st.markdown(f"**{reward['cost']} WP**")
                            
                            is_disabled = total_points < reward['cost']
                            
                            # --- [핵심 수정] 버튼 클릭 시 /redeem_reward API 호출 ---
                            if st.button("교환 신청", key=key, use_container_width=True, disabled=is_disabled):
                                try:
                                    # 새로 추가한 엔드포인트에 'cost'를 json으로 전송
                                    response = requests.post(f"{FLASK_SERVER_URL}/redeem_reward", 
                                                             json={"cost": reward['cost']})
                                    
                                    if response.status_code == 200:
                                        result = response.json()
                                        st.success(f"'{reward['name']}' 교환 신청 완료!")
                                        st.toast(f"남은 포인트: {result.get('new_points', 'N/A')} WP")
                                        
                                        # (중요) 페이지를 새로고침하여 상단의 WP 값을 갱신
                                        st.rerun() 
                                    else:
                                        # (예: 포인트 부족, 서버 오류 등)
                                        result = response.json()
                                        st.error(f"⚠️ 교환 실패: {result.get('message', '서버 오류')}")
                                
                                except Exception as e:
                                    st.error(f"⚠️ 서버 연결 오류: {e}")
                            # --- [수정 완료] ---
                            
                            if is_disabled:
                                st.caption(f"{reward['cost'] - total_points} WP 부족")

            # --- 리더보드 탭 ---
            with tab_leaderboard:
                st.subheader("🥇 이달의 웰니스 리더보드 (예시)")
                st.info("이 기능은 기업/단체 관리자 모듈과 연동 후 활성화됩니다.")
                st.markdown("""
                | 순위 | 사용자 | 포인트 |
                | :---: | :---: | :---: |
                | 1. 🥇 | User-A (경영지원) | 15,200 WP |
                | 2. 🥈 | User-B (개발팀) | 13,100 WP |
                | 3. 🥉 | User-C (디자인팀) | 11,500 WP |
                | ... | ... | ... |
                | **8.** | **(나)** | **{points} WP** |
                """.format(points=total_points))
                
            # --- 나의 배지 탭 (UI 단순화) ---
            with tab_badges:
                st.subheader("🏆 나의 배지 컬렉션")
                st.caption("획득한 배지는 컬러로 표시됩니다.")

                badge_info = {
                    'first_steps': {'title': '첫 걸음', 'description': '좋은 자세 10분 달성', 'icon': '👶'},
                    'streak_master': {'title': '연속 달인', 'description': '30분 연속 좋은 자세', 'icon': '🔥'},
                    'posture_pro': {'title': '자세 전문가', 'description': '총 100분 좋은 자세', 'icon': '👨‍⚕️'},
                    'perfect_day': {'title': '완벽한 하루', 'description': '8시간 연속 좋은 자세', 'icon': '🌟'}
                }
                earned_badges = user_stats.get('badges', [])
                
                cols = st.columns(len(badge_info))
                
                for i, (badge_name, info) in enumerate(badge_info.items()):
                    with cols[i]:
                        if badge_name in earned_badges:
                            # 획득한 배지 (컬러)
                            st.markdown(f"""
                            <div style="text_align: center; padding: 10px; border: 2px solid #4CAF50; border_radius: 10px; background_color: #f0f8f0; height: 150px;">
                                <h3>{info['icon']}</h3>
                                <h4>{info['title']}</h4>
                                <p style="font_size: 12px;">{info['description']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            # 획득하지 못한 배지 (회색)
                            st.markdown(f"""
                            <div style="text_align: center; padding: 10px; border: 2px solid #ccc; border_radius: 10px; background_color: #f5f5f5; height: 150px; opacity: 0.6;">
                                <h3>{info['icon']}</h3>
                                <h4 style="color: #666;">{info['title']}</h4>
                                <p style="font_size: 12px; color: #666;">{info['description']}</p>
                            </div>
                            """, unsafe_allow_html=True)

        else:
            st.error("⚠️ 웰니스 데이터 로딩 실패. Flask 서버가 실행 중인지 확인하세요.")
            
    except Exception as e:
        st.error(f"⚠️ 웰니스 통계 로드 오류: {e}")
        st.info("Flask 서버를 먼저 실행해주세요. (`streaming_server.py`)")

# --- 스마트 알림 관리 탭 ---
# [교체] tab7 블록 전체
# --- 스마트 알림 관리 탭 ---
with tab7:
    st.header("🔔 스마트 알림 관리")
    st.info("💡 AI가 분석한 당신의 자세 패턴을 바탕으로 맞춤형 알림을 설정하고 관리하세요. (패턴 분석은 항상 '최근 7일' 데이터를 기준으로 합니다)")
    
    col1, col2 = st.columns([0.6, 0.4])
    
    with col1:
        st.subheader("📊 자세 패턴 분석 (최근 7일)")
        if st.button("🔍 패턴 분석 실행", use_container_width=True):
            try:
                # [수정] tab7은 날짜 필터 없이 항상 '최근 7일' 기준으로 분석
                # (서버에서 날짜값이 안 넘어오면 기본값 7일로 작동)
                response = requests.get(f"{FLASK_SERVER_URL}/analyze_patterns") 
                
                if response.status_code == 200:
                    worst_hours = response.json()['worst_hours']
                    if worst_hours:
                        st.success("✅ 최근 7일간의 패턴 분석이 완료되었습니다!")
                        st.subheader("🚨 문제가 많은 시간대 (최근 7일)")
                        for i, (hour, frequency) in enumerate(worst_hours):
                            st.write(f"{i+1}. **{hour}시**: {frequency}회 나쁜 자세 감지")
                        
                        if worst_hours:
                            hours_data = []
                            for hour, freq in worst_hours:
                                hours_data.append({'시간대': f"{hour}시", '나쁜 자세 횟수': freq})
                            df_hours = pd.DataFrame(hours_data)
                            fig = px.bar(df_hours, x='시간대', y='나쁜 자세 횟수', title='시간대별 나쁜 자세 발생 빈도 (최근 7일)', color='나쁜 자세 횟수', color_continuous_scale='Reds')
                            fig.update_layout(width=500, height=300)
                            st.plotly_chart(fig, config={'displayModeBar': True})
                    else:
                        st.info("ℹ️ 아직 충분한 데이터가 없습니다. (최근 7일)")
                else:
                    st.error("⚠️ 패턴 분석에 실패했습니다.")
            except Exception as e:
                st.error(f"⚠️ 패턴 분석 오류: {e}")
    
    with col2:
        st.subheader("⚙️ 스마트 알림 설정 (최근 7일 기준)")
        if st.button("🔔 스마트 알림 생성", use_container_width=True, help="최근 7일 데이터를 기반으로 알림을 생성합니다."):
            try:
                # [수정] 알림 생성도 '최근 7일' 고정 (서버에서 자동 처리)
                response = requests.post(f"{FLASK_SERVER_URL}/create_smart_notifications")
                if response.status_code == 200:
                    st.success("✅ 스마트 알림이 생성되었습니다!"); st.rerun()
                else:
                    st.error("⚠️ 스마트 알림 생성에 실패했습니다.")
            except Exception as e:
                st.error(f"⚠️ 스마트 알림 생성 오류: {e}")
        
        st.divider()
        
        st.subheader("📋 활성화된 알림")
        try:
            response = requests.get(f"{FLASK_SERVER_URL}/get_smart_notifications")
            if response.status_code == 200:
                notifications = response.json()['notifications']
                if notifications:
                    for notif in notifications:
                        st.write(f"⏰ **{notif['trigger_time']}**: {notif['type']}")
                        if notif['last_triggered']: st.caption(f"마지막 실행: {notif['last_triggered']}")
                        else: st.caption("아직 실행되지 않음")
                else:
                    st.info("활성화된 알림이 없습니다.")
            else:
                st.error("⚠️ 알림 목록을 불러올 수 없습니다.")
        except Exception as e:
            st.error(f"⚠️ 알림 목록 로드 오류: {e}")