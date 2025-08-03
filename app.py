import pandas as pd
import streamlit as st
import re
import warnings
from datetime import datetime
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

# 환경 변수 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")  # 경고 메시지 무시

# To-Do 쿼리 파싱 함수
## 자연어 쿼리에서 날짜 / 시각 / 키워드 추출 (todo.csv용)
def parse_todo_query(query):

    today = datetime.now()
    date, hour, keyword = None, None, None

    # 날짜 추출 패턴: "2025-08-02", "8월 2일", "8.2", "8/2" 등
    dt = re.search(r'(20\d{2})[- /.년]+(\d{1,2})[- /.월]+(\d{1,2})', query)
    if dt:
        year, month, day = dt.groups()
        date = f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
    else:
        dt2 = re.search(r'(\d{1,2})[월\.]?\s*(\d{1,2})[일]?', query)
        # 현재 연도를 사용하여 날짜 생성
        if dt2:
            month, day = dt2.groups()
            date = f"{today.year}-{int(month):02d}-{int(day):02d}"

    # 시간 추출 예시: "10시", "10시 30분", "14:00" 등만 잡기
    tm = re.search(r'(\d{1,2})시', query) or re.search(r'(\d{1,2}):\d{2}', query)
    if tm:
        try:
            _h = int(tm.group(1))
            hour = _h if 0 <= _h <= 23 else None
        except:
            hour = None

    # 키워드 추출 (할 일 주제)
    TASK_KEYWORDS = [
        "운동", "스터디", "점심", "조깅", "강의", "복습", "모임", "과제", 
        "세미나", "회의", "발표", "식사", "외식", "논문", "AI", "인턴", "프로젝트",
        "시험", "시험공부", "학습", "독서", "여행", "휴식", "정리", "청소",
        "쇼핑", "장보기", "약속", "친구", "가족", "영화", "드라마",
        "게임", "취미", "취업", "자격증", "자기계발", "자기개발", "자기관리",
    ]
    for key in TASK_KEYWORDS:
        if key in query:
            keyword = key
            break

    return date, hour, keyword


# SnoWe 게시글 쿼리 파싱 함수
## 자연어 쿼리에서 키워드 (장학, 학사, 공지 등 주제 키워드) 및 날짜 추출 (snowe_article.csv용)
## 추후 웹 스크래핑 확장 예정
def parse_article_query(query):

    from datetime import datetime
    keyword, date = None, None
    CATEGORY_KEYWORDS = [
        "장학", "휴학", "복학", "계절학기", 
        "학점교류", "현장실습", "공모전", "AI", 
        "교육", "워크숍", "강의", "등록", "인공지능", "빅데이터",
        "학술", "연구", "학회", "학술대회",
        "졸업", "논문", "인턴", "성평등", "폭력예방",
        "학생회", "모집", "행사", "세미나",
        "학사", "공지", "학적", "수업", "시험",
        "학위", "입학", "전형", "장학금", "장학제도",
        "학술제", "학술행사", "학생복지", "학생지원",
        "학생회", "학생활동", "학생자치", "학생회비"
    ]

    for key in CATEGORY_KEYWORDS:
        if key in query:
            keyword = key
            break

    # 날짜 추출 예시: "8월 2일", "2025년 8월 1일"
    dt = re.search(r'(20\d{2})[- /.년]+(\d{1,2})[- /.월]+(\d{1,2})', query)
    if dt:
        year, month, day = dt.groups()
        date = f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
    else:
        dt2 = re.search(r'(\d{1,2})[월./\- ]+(\d{1,2})[일]?', query)
        if dt2:
            today = datetime.now()
            month, day = dt2.groups()
            date = f"{today.year}-{int(month):02d}-{int(day):02d}"

    return keyword, date


# To-Do 전용 RAG 실행 함수
def execute_todo_query(query, embeddings, todo_df):
    date, hour, keyword = parse_todo_query(query)
    df = todo_df.copy()

    if date:
        df = df[df["date"] == date]

    if keyword:
        df = df[df["description"].str.contains(keyword, na=False) | df["title"].str.contains(keyword, na=False)]

    if hour is not None:
        df = df[df["time"].apply(lambda t: int(t.split(":")[0]) == hour)]

    # 디버깅 로그 출력
    print("🔍 [TODO QUERY PARSING RESULTS]")
    print("query input >>", query)
    print("parsed keyword:", keyword)
    print("parsed date:", date)
    print("parsed hour:", hour)
    print("filtered rows:", df.shape[0])
    print()

    if df.empty:
        return [{"no_result": "🔎 조건에 맞는 일정이 없습니다."}]

    # 날짜 위주 검색의 경우 -> 벡터 검색 없이 해당 날짜의 일정 모두 리턴
    if keyword is None:
        matched_results = []
        seen_keys = set()
        for _, row in df.iterrows():
            key = f"{row['title']}@{row['date']}@{row['time']}"
            if key not in seen_keys:
                matched_results.append({
                    "date": row['date'],
                    "time": row['time'],
                    "title": row['title'],
                    "description": row['description'],
                    "location": row['location']
                })
                seen_keys.add(key)

        return matched_results
    
    # 키워드 또는 날짜가 있는데 hour가 없는 경우 -> 바로 출력
    elif hour is None:
        matched_results = []
        seen_keys = set()
        for _, row in df.iterrows():
            key = f"{row['title']}@{row['date']}@{row['time']}"
            if key not in seen_keys:
                matched_results.append({
                    "date": row['date'],
                    "time": row['time'],
                    "title": row['title'],
                    "description": row['description'],
                    "location": row['location']
                })
                seen_keys.add(key)
        return matched_results

    # 벡터 유사도 검색 수행
    def format_todo(row):
        return f"{row['date']} {row['time']} | {row['title']} | {row['description']} | {row['location']}"

    texts = df.apply(format_todo, axis=1).tolist()
    sub_db = Chroma.from_texts(texts, embeddings)
    results = sub_db.similarity_search(query, k=min(5, len(texts)))

    # 구조화된 데이터 반환
    matched_results = []
    seen_keys = set()   # 중복 방지
    for doc in results:
        for _, row in df.iterrows():
            combined = f"{row['date']} {row['time']} | {row['title']} | {row['description']} | {row['location']}"
            unique_key = f"{row['title']}@{row['date']}@{row['time']}"  # title + date + time 조합으로 key 생성 -> 중복 체크
            if combined in doc.page_content and unique_key not in seen_keys:
                matched_results.append({
                    "date": row['date'],
                    "time": row['time'],
                    "title": row['title'],
                    "description": row['description'],
                    "location": row['location']
                })
                seen_keys.add(unique_key)   # 중복 방지용 체크 -> 고유 키로 확인
                break   # 한 row 매치되면 중복 방지용으로 break

    return matched_results


# SnoWe Announcement Article 전용 RAG 실행 함수
def execute_article_query(query, embeddings, article_df):
    keyword, date = parse_article_query(query)
    df = article_df.copy()

    if keyword:
        df = df[
            df["title"].str.contains(keyword, case=False, na=False) | 
            df["category"].str.contains(keyword, case=False, na=False)
        ]

    if date:
        df = df[
            (df["start_date"] <= date) &
            (df["end_date"] >= date)
        ]

    # 디버깅 로그 출력
    print("🔍 [ARTICLE QUERY PARSING RESULTS]")
    print("query input >>", query)
    print("parsed keyword:", keyword)
    print("parsed date:", date)
    print("filtered rows:", df.shape[0])
    print()

    if df.empty:
        return [{"no_result": "🔎 조건에 맞는 일정이 없습니다."}]

    # 단순 키워드 - 벡터 검색 생략 조건
    SIMPLE_KEYWORDS = {"모집", "장학", "휴학", "복학", "논문", "교육", "공모전", "현장실습"}
    
    # filtered rows가 5개 이하, 쿼리 조건이 매우 단순한 경우(단순 키워드 검색, 1-3단어 이내 검색) -> 벡터 검색 없이 바로 출력
    if (df.shape[0] <= 5
        or (keyword in SIMPLE_KEYWORDS)
        or (len(query.split()) <= 3 and date is None)):
        matched_results = []
        seen_keys = set()
        for _, row in df.iterrows():
            key = f"{row['title']}@{row['start_date']}@{row['end_date']}"
            if key not in seen_keys:
                matched_results.append({
                    'title': row['title'],
                    'url': row['url'],
                    'date_range': f"{row['start_date']} ~ {row['end_date']}",
                    'category': row['category']
                })
                seen_keys.add(key)
        return matched_results

    # 벡터 유사도 검색 수행
    texts = df.apply(lambda row: f"{row['category']} | {row['title']} | {row['url']}", axis=1).tolist()
    sub_db = Chroma.from_texts(texts, embeddings)
    results = sub_db.similarity_search(query, k=min(5, len(texts)))

    matched_results = []
    seen_keys = set()  # 중복 방지용
    for doc in results:
        for idx, row in df.iterrows():
            unique_key = f"{row['title']}@{row['start_date']}@{row['end_date']}"  # title + start_date + end_date 조합으로 key 생성
            if row['title'] in doc.page_content and unique_key not in seen_keys:
                matched_results.append({
                    'title': row['title'],
                    'url': row['url'],  # 하이퍼링크로 사용
                    'date_range': f"{row['start_date']} ~ {row['end_date']}",
                    'category': row['category']
                })
                seen_keys.add(unique_key)
                break

    return matched_results


# Streamlit 앱 메인 함수
def main():
    st.set_page_config(page_title="CampusMate-RAG", layout='wide')
    st.title("🦾 CampusMate-RAG")
    st.subheader("\"당신만을 위한 To-Do & 학교 공지사항 RAG 검색 시스템\"")

    # CSV 불러오기 (지정 경로)
    todo_df = pd.read_csv("./data/todo.csv")
    article_df = pd.read_csv("./data/snowe_article.csv")

    for df in [todo_df, article_df]:
        df.columns = df.columns.str.strip()
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # 파일 로딩 직후 - 날짜 컬럼 처리
    todo_df['date'] = todo_df['date'].astype(str).str.strip()

    # datetime 컬럼 처리
    article_df["start_date"] = pd.to_datetime(article_df["start_date"])
    article_df["end_date"] = pd.to_datetime(article_df["end_date"])

    # 임베딩 모델 초기화
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sbert-sts",   # 로그인/토큰 없이 사용 가능(공개 모델)
        model_kwargs={"device": "cpu"}
    )
    
    if "current_tab" not in st.session_state:
        st.session_state.current_tab = "todo"

    # Streamlit UI 탭 구분
    tab1, tab2 = st.tabs(["🗓 To-Do/일정 관리", "📢 학교 공지 게시글 검색"])

    with tab1:
        if "query_todo" not in st.session_state:
            st.session_state.query_todo = ""
        st.subheader("🗓 To-Do/일정 질의하기")
        st.markdown("💡 *예: '8월 2일에 뭐 있어?'*")
        query_todo = st.text_input("🔍 질문을 입력하세요:", key="query_todo")

        # 예시 버튼
        with st.expander("📎 예시 질문 보기"):
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗓️ 8월 2일 일정 뭐 있어?", key="todo_example_1"):
                    query_todo = "2025년 8월 2일 일정 뭐 있어?"
                if st.button("🏃‍♀️ 운동 언제해?", key="todo_example_2"):
                    query_todo = "운동 있는 날은 언제야?"
            with col2:
                if st.button("🍽️ 점심 약속 있는 날 알려줘", key="todo_example_3"):
                    query_todo = "점심 약속 있는 날 알려줘"
                if st.button("📄 스터디 모임 언제 있었더라?", key="todo_example_4"):
                    query_todo = "스터디 모임이 있는 날은 언제야?"

        # 사용자 쿼리 입력
        if query_todo:
            results = execute_todo_query(query_todo, embeddings, todo_df)

            parsed_date, parsed_hour, parsed_keyword = parse_todo_query(query_todo)
            if parsed_date and not parsed_keyword:
                st.info(f"📅 {parsed_date}에 등록된 모든 일정을 보여드릴게요!")

            for idx, r in enumerate(results, 1):
                # 결과가 dict가 아니면(예: str), 안내문 표시!
                if not isinstance(r, dict):
                    st.info(str(r))
                    continue

                # 결과가 없는 경우
                if "no_result" in r:
                    st.info(r["no_result"])
                    continue

                st.markdown(f"""
                    <div style="border:1px solid #f0d8d8; padding:12px; border-radius:8px; margin-bottom:10px; background-color:#fff8f8">
                    <b>🗓 일정 {idx}</b>  
                    <br>📅 날짜: {r['date']}  
                    <br>⏰ 시간: {r['time']}  
                    <br>📌 제목: {r['title']}  
                    <br>🧾 설명: {r['description']}  
                    <br>📍 장소: {r['location']}
                    </div>
                """, unsafe_allow_html=True)

    with tab2:
        st.subheader("📢 학교 공지사항 질의하기")
        st.markdown("💡 *예: '장학금 관련 공지 알려줘', '논문 제출 언제까지야?'*")
        query_article = st.text_input("🔍 질문을 입력하세요:", key="query_article")

        # 예시 버튼
        with st.expander("📎 예시 질문 보기"):
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗓️ 2025년도 2학기 복학 신청 일정 언제야?", key="article_example_1"):
                    query_article = "2025년도 2학기 복학 신청 일정 언제야?"
                if st.button("✅ 현재 모집하는 행사는 어떤게 있어?", key="article_example_2"):
                    query_article = "현재 모집하는 행사는 어떤게 있어?"
            with col2:
                if st.button("🎓 장학금 공지 알려줘", key="article_example_3"):
                    query_article = "장학금 관련 공지 알려줘"
                if st.button("📄 합격자 발표 나온거 있어?", key="article_example_4"):
                    query_article = "합격자 발표 나온거 있어?"

        # 사용자 쿼리 입력
        if query_article:
            results = execute_article_query(query_article, embeddings, article_df)
            for idx, r in enumerate(results, 1):
                # 결과가 dict가 아니면(예: str), 안내문 표시!
                if not isinstance(r, dict):
                    st.info(str(r))
                    continue

                # 결과가 없는 경우
                if "no_result" in r:
                    st.info(r["no_result"])
                    continue

                st.markdown(f"""
                    <div style="border:1px solid #d8e2f0; padding:12px; border-radius:8px; margin-bottom:10px; background-color:#eef4fb">
                    <b>📢 공지 {idx}</b>  
                    <br>🗂 카테고리: {r['category']}  
                    <br>📅 기간: {r['date_range']}  
                    <br>🔗 제목: <a href="{r['url']}" target="_blank">{r['title']}</a>
                    </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()