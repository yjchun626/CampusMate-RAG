# CampusMate-RAG: Personalized ToDo & School Announcement Search System

**Streamlit 기반 자연어 질의로 ‘개인 일정’과 ‘학교 공지’를 검색할 수 있는 AI RAG 챗봇**

---

## 🌟 서비스 데모

- **배포된 [Streamlit App](https://campusmate-rag.streamlit.app/)으로 실시간 체험**

---

## ✨ 주요 기능

- 자연어로 일정/공지(예: “8월 3일 운동 뭐 있어?”, “장학금 모집 안내”) 검색
- 날짜/키워드/시각 기반 파싱 & 커스텀 RAG
- 중복 결과 필터링 & 카드형 UI
- Pandas + 한국어 임베딩 챗봇 + Streamlit
- 데이터셋 `.csv` 구조 지원

---

## 🛠️ 기술스택

- **Python 3.12+**
- **Streamlit** (UI/배포)
- **pandas, re, datetime**
- **HuggingFace Sentence-Transformers (`jhgan/ko-sbert-sts`)**
- **Langchain, Chroma**

---

## 💾 데이터 & 폴더 구조
```plaintext
project-root/
├── app.py (메인 실행 파일)
├── data/
│   ├── todo.csv       # 일정/할일 데이터
│   └── article.csv    # 공지 게시글 데이터
├── README.md
└── requirements.txt
```

---

## 📋 시스템 아키텍처 요약

1. 사용자 자연어 쿼리 입력
2. 날짜/키워드/시각 파싱(정규식)
3. pandas로 candidate 필터링
4. 필요시 임베딩 기반 RAG 또는 필터 결과만 사용
5. 결과 데이터값 중복 제거 후 카드 UI로 결과 표시

---

## 💻 설치 및 실행 안내

1. **필수 환경**
   - Python >= 3.10 (추천 3.12)
   - pip 업데이트

2. **의존 패키지 설치**
```python
pip install -r requirements.txt
```

3. **데이터 파일 준비**
- data/todo.csv, data/article.csv에 샘플 데이터 추가
- Note: 추후 데이터 파일 임의 지정 필요 없이 웹 스크래핑을 통해 데이터셋 얻는 기능 구현 예정

4. **Streamlit 실행**
```python
streamlit run app.py
```
- 실행 후 주소(예: `http://localhost:8501`)에서 확인
