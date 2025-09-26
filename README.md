# Velog
## **1개월차: ML/DL 기초 & 실무 응용**

**🎯 목표**: 기초 ML/DL 개념을 익히고, 기업 문제(예측·분류)에 적용 가능한 기반 확보

✅ **블로그 포스트 시리즈**

1. 머신러닝 개요: 지도/비지도/강화학습 + 비즈니스 문제 정의
2. 데이터 전처리 & Feature Engineering 실무 팁 (결측치, 인코딩, 스케일링)
3. 선형회귀·로지스틱회귀·SVM 실습 + 평가 지표(AUC, F1, Confusion Matrix)
4. 앙상블 모델 (Random Forest, XGBoost, LightGBM) + Kaggle 실습 기록
5. 딥러닝 기초: 퍼셉트론, MLP, Backpropagation 시각화 설명
6. PyTorch로 MNIST 분류기 만들기
7. CNN (CIFAR-10/100 이미지 분류 실습)
8. RNN·LSTM (텍스트/시계열 처리 기초)

📌 **논문 리뷰**

- LeCun (1998) CNN-LeNet
- Hochreiter & Schmidhuber (1997) LSTM
- Chen & Guestrin (2016) XGBoost
- Kingma & Ba (2014) Adam Optimizer

---

## **🔹 2개월차: NLP & Transformer 핵심**

**🎯 목표**: NLP 전처리 → 임베딩 → Transformer·BERT/GPT 기반 모델 이해

✅ **블로그 포스트 시리즈**

1. NLP 기초: 토큰화(WordPiece, SentencePiece) + 임베딩(Word2Vec, GloVe)
2. 텍스트 분류 실습 (감정 분석, 형태소 분석기 활용)
3. Seq2Seq와 Attention 메커니즘 (번역기 구조 해설)
4. Transformer 논문 리뷰 (Attention is All You Need)
5. Transformer 구조 완벽 해설 (MHA, FFN, Residual, LayerNorm)
6. BERT 활용 (문장 분류·질의응답 실습)
7. GPT 발전사 (GPT-2~3, In-context Learning)
8. 한국어 LLM (KoBERT, KoGPT, KULLM, KLUE) 실습기

📌 **논문 리뷰**

- Mikolov et al. (2013) Word2Vec
- Vaswani et al. (2017) Attention is All You Need
- Devlin et al. (2019) BERT
- Brown et al. (2020) GPT-3
- Park et al. (2021) KLUE

---

## **🔹 3개월차: LLM 실전 & RAG 프로젝트**

**🎯 목표**: LLM 파인튜닝, RAG, 서비스형 프로젝트 구현 및 배포

✅ **블로그 포스트 시리즈**

1. LLM 파인튜닝 전략: Fine-tuning vs LoRA vs QLoRA 비교
2. Hugging Face Trainer로 LoRA 파인튜닝 실습
3. RLHF (InstructGPT) & DPO (Direct Preference Optimization) 개념
4. RAG (Retrieval-Augmented Generation) 구조 & 필요성
5. FAISS/Pinecone로 벡터 검색 구현
6. LangChain & LlamaIndex로 문서 검색 챗봇 만들기
7. 프로젝트①: 요리 레시피 챗봇 (멀티모달: CV+NLP)
8. 프로젝트②: 뉴스 요약 서비스 (KoBART/PEGASUS 활용)
9. 프로젝트③: 생성형 AI 탐지기 (DetectGPT + KoBERT)
10. FastAPI + Docker로 모델 배포하기
11. CI/CD 적용 경험 (GitHub Actions, Docker Hub 배포)

📌 **논문 리뷰**

- Hu et al. (2021) LoRA
- Dettmers et al. (2023) QLoRA
- Ouyang et al. (2022) InstructGPT (RLHF)
- Rafailov et al. (2023) DPO
- Lewis et al. (2020) RAG
- Reimers & Gurevych (2019) Sentence-BERT

---

## **🔹 4개월차: AI Agent & 최신 트렌드**

**🎯 목표**: LLM → Agent 확장, 툴 사용/계획/자율 실행까지 경험

✅ **블로그 포스트 시리즈**

1. AI Agent 개요: LLM → Action → Observation → Re-Plan 루프
2. Prompt Engineering 고급 기법 (Chain-of-Thought, Self-Consistency)
3. ReAct 프레임워크 (Reasoning + Acting) 논문 리뷰 및 실습
4. Toolformer: LLM이 스스로 API 툴 호출 학습
5. LangGraph 기본 구조 (노드·엣지 기반 워크플로우)
6. LangGraph로 “계획→툴 호출→검증→재계획” 루프 구현
7. 캘린더/메일 API 연동 Agent 시나리오 구축
8. 장애주입 테스트 (툴 실패 시 복구 경로 설계)
9. 시나리오 20개 자동 평가 스크립트 작성
10. 최신 논문 리뷰 & 트렌드 정리

📌 **논문 리뷰**

- Wei et al. (2022) Chain-of-Thought
- Yao et al. (2022) ReAct
- Schick et al. (2023) Toolformer
- LangGraph 공식 레퍼런스

---

## **✅ 최종 정리**

- **1개월차**: ML/DL 기초 → 회사에서 자주 쓰는 예측/분류 기반 문제 해결 능력
- **2개월차**: NLP/Transformer → BERT/GPT 이해 및 서비스 적용
- **3개월차**: LLM/RAG → 챗봇, 요약기, 탐지기 프로젝트 + 배포 경험
- **4개월차**: Agent → LangGraph/툴 연동 + 자율 실행 AI Agent 구축
