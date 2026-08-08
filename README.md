# Fengshui Chatbot — Vạn An Group

Hệ thống **chatbot tư vấn sản phẩm phong thủy** (multi-agent) + **pipeline xử lý dữ liệu sản phẩm** cho shop Vạn An Group.

Repo: [AnhDT1704/Fengshui_chatbot](https://github.com/AnhDT1704/Fengshui_chatbot)

---

## Tổng quan

Dự án gồm **hai tầng**:

| Tầng | Vai trò |
|------|---------|
| **Data pipeline** (root) | Parse / extract metadata / embedding → ghi **PostgreSQL** + index **OpenSearch** |
| **Chatbot** (`langraph pipeline/`) | Multi-agent **LangGraph** + FastAPI UI, tra DB/OS, tư vấn mệnh–size, CSKH |

Stack chính: **LangGraph · Gemini · PostgreSQL · OpenSearch · FastAPI · SigLIP (visual search)**.

Model finetune (tuỳ chọn, qua ngrok/Colab):

- **VLM (ảnh)** — `FINETUNE_API_URL` → nhận diện SP từ ảnh khách  
- **Phong thủy (text)** — `FENGSHUI_API_URL` → năm sinh / can chi / cổ tay → mệnh & size (fallback code nếu tắt)

---

## Luồng chatbot (1 request)

```
Khách (UI / API)
       │
       │  POST /chat  hoặc  POST /chat/image  (≤5 ảnh)
       ▼
  FastAPI  (langraph pipeline/api.py)
       │
       │  nạp memory (conversation_log) + session status
       ▼
  graph.chat() / chat_with_image()
       │
       ▼
  ┌─────────────────────┐
  │  supervisor_node    │  lập PLAN: 1 hoặc nhiều agent (chuỗi "A -> B")
  └──────────┬──────────┘
             │
     route theo plan[step]
             │
    ┌────────┼────────────────┬──────────────────┐
    ▼        ▼                ▼                  ▼
 small_talk  knowledge_base  skills_agent   order_support
 (xã giao)   (SP + ảnh +     (tính size /   (CSKH + escalate
              mệnh)           số hạt)        chủ shop)
    │             │                │                  │
    └─────────────┴────────────────┴──────────────────┘
             │
             │  (có thể chạy 2 agent: vd skills -> KB)
             ▼
  post-process: verify chống bịa SP, CTA Shopee (giá/KM),
                handoff pending_admin nếu escalate
             │
             ▼
  log conversation + JSON/stream về khách
```

### Agent & trách nhiệm

| Agent | Khi nào dùng |
|-------|----------------|
| **small_talk** | Chào, cảm ơn, tạm biệt, chat xã giao |
| **knowledge_base_agent** | Có bán SP không, lọc/so sánh SP, **ảnh**, tư vấn **mệnh/năm sinh**, HDSD, số hạt mặc định theo size |
| **skills_agent** | Có **số đo cổ tay** / vóc dáng → tính **size li + số hạt** (Sinh–Lão–Bệnh–Tử), tư vấn quà |
| **order_support_agent** | Bảo hành, đổi trả, KM; giao hàng / khiếu nại / dịch vụ phụ → **escalate chủ shop** |
| **off_platform_policy** | Xin SĐT/Zalo/địa chỉ shop hoặc giao dịch ngoài Shopee → **câu trả lời cố định** (regex + policy) |

Supervisor có thể **xếp chuỗi**, ví dụ:

- Ảnh + hỏi size: `skills_agent -> knowledge_base_agent`
- Chỉ tra SP / ảnh: `knowledge_base_agent`
- Chỉ tính cổ tay: `skills_agent`

### Knowledge base — công cụ chính

- `keyword_search_tool` / `semantic_search_tool` / `filter_search_tool` (OpenSearch + Postgres)
- `image_search_tool` (SigLIP embedding → kNN)
- `get_product_detail_tool`, `product_care_tool`, …
- `fengshui_advisor_tool` / size (model FT nếu bật `FENGSHUI_API_URL`, không thì code chu kỳ 60 năm)
- Finetune VLM (`FINETUNE_API_URL`) khi khách gửi ảnh

### Tính năng vận hành

- **Auth** (user / admin), lịch sử phiên theo user  
- **Handoff** chủ shop (`pending_admin`) khi escalate  
- **Admin**: nạp Excel giá–tồn / khuyến mãi, cài đặt runtime  
- **UI** tĩnh: `langraph pipeline/static/index.html` (cổng 8000)  
- **Streamlit** demo: `streamlit_app.py`  
- **SSE progress** khi chat kèm ảnh (`/chat/image/stream`)

---

## Cấu trúc thư mục (rút gọn)

```
.
├── docker-compose.yml          # postgres, pgadmin, opensearch, dashboards, chatbot
├── Dockerfile                  # image FastAPI (workdir: langraph pipeline/)
├── requirements.txt
├── .env.example
├── config.py / models.py       # cấu hình + SQLAlchemy
├── db_service.py               # PostgreSQL
├── opensearch_service.py       # index + search
├── embedding_service.py        # text-embedding-3-small (OpenRouter)
├── pipeline.py                 # CLI pipeline dữ liệu SP
├── product_parser.py
├── metadata_extractor.py
├── chunk_builder.py
├── migrations/                 # SQL/Python migration chatbot + schema
├── data/                       # file .txt sản phẩm gốc
├── langraph pipeline/          # CHATBOT
│   ├── api.py                  # FastAPI
│   ├── graph.py                # LangGraph wiring + chat()
│   ├── supervisor_agent.py     # routing / plan
│   ├── knowledge_base_agent.py
│   ├── skills_agent.py
│   ├── order_support_agent.py
│   ├── memory.py / auth.py
│   ├── fengshui_finetune_client.py
│   ├── image_embedding.py      # SigLIP
│   ├── response_verifier.py
│   ├── static/index.html
│   └── streamlit_app.py
└── dataset_fengshui_cot/       # dataset fine-tune mệnh/size (JSONL + CoT)
```

---

## Setup nhanh

### 1. Clone & env

```bash
git clone https://github.com/AnhDT1704/Fengshui_chatbot.git
cd Fengshui_chatbot

cp .env.example .env
# Bắt buộc: GOOGLE_API_KEY1=...  (Gemini)
# Khuyến nghị: OPENROUTER_API_KEY=...  (embedding pipeline)
# Tuỳ chọn: SERPAPI_KEY, FINETUNE_API_URL, FENGSHUI_API_URL
```

### 2. Hạ tầng + chatbot (Docker)

```bash
docker compose up -d
```

| Service | URL / port |
|---------|------------|
| Chatbot API + UI | http://localhost:8000 |
| Health | http://localhost:8000/health |
| OpenSearch | http://localhost:9200 |
| OpenSearch Dashboards | http://localhost:5601 |
| PostgreSQL | localhost:5432 |
| pgAdmin | http://localhost:8080 |

Trong container, `PG_HOST=postgres`, `OS_HOST=opensearch` (override `.env` localhost).

```bash
docker compose logs -f chatbot
docker compose restart chatbot
docker compose build chatbot          # khi đổi requirements.txt
docker compose up -d postgres opensearch   # chỉ hạ tầng
```

Code host bind-mount `.:/app` + uvicorn `--reload` (Windows: `WATCHFILES_FORCE_POLLING=true`).

### 3. Chạy local (venv) — không Docker chatbot

```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
pip install -r requirements.txt

# Cần postgres + opensearch đang chạy (compose hoặc cài sẵn)
cd "langraph pipeline"
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Demo Streamlit:

```bash
cd "langraph pipeline"
streamlit run streamlit_app.py
```

### 4. Nạp / cập nhật dữ liệu sản phẩm

```bash
# Pipeline đầy đủ: parse → metadata → chunk → embed → Postgres + OpenSearch
python pipeline.py

# Từng bước
python pipeline.py --steps 1,2,3 --save-json
python pipeline.py --steps 4,5,6,7
python pipeline.py --reset-db
```

Migration schema chatbot (auth, handoff, product ref, …): thư mục `migrations/`.

Chủ shop có thể nạp **Excel giá/tồn** và **khuyến mãi** qua UI admin (template + import API).

---

## API chính (chatbot)

| Method | Path | Mô tả |
|--------|------|--------|
| GET | `/` | UI tĩnh |
| GET | `/health` | Healthcheck |
| POST | `/chat` | Chat text `{ session_id, message }` |
| POST | `/chat/image` | Chat + ảnh (multipart / base64) |
| POST | `/chat/image/stream` | Chat ảnh + SSE progress |
| GET/DELETE | `/history/{session_id}` | Lịch sử phiên |
| POST | `/auth/login`, `/auth/register` | Auth |
| GET | `/sessions` | Phiên theo user |
| GET | `/admin/handoffs` | Phiên chờ chủ shop |
| POST | `/admin/reply` | Chủ shop trả lời |
| POST | `/admin/import/products` | Nạp Excel giá/tồn |
| POST | `/admin/import/promotions` | Nạp Excel KM |
| GET | `/escalations` | Hàng đợi escalate |

---

## Biến môi trường quan trọng

Xem đầy đủ trong `.env.example`.

| Biến | Ý nghĩa |
|------|---------|
| `GOOGLE_API_KEY1..N` | Gemini (bắt buộc cho chatbot; rotate quota) |
| `CHATBOT_MODEL` | Mặc định `gemini-2.5-flash` |
| `CHATBOT_MEMORY_LIMIT` | Số lượt nạp context (`0` = tắt memory) |
| `CHATBOT_MAX_IMAGES` | Số ảnh / lượt (mặc định 5) |
| `OPENROUTER_API_KEY` | Embedding pipeline |
| `PG_*` / `OS_*` | Postgres / OpenSearch |
| `FINETUNE_API_URL` | VLM nhận diện ảnh (ngrok) — trống = tắt |
| `FENGSHUI_API_URL` | Model mệnh/size (ngrok) — trống = fallback code |
| `SERPAPI_KEY` | web search (tuỳ chọn) |

---

## Pipeline dữ liệu (tóm tắt)

```
Raw .txt (--N--)
    → product_parser
    → metadata_extractor
    → chunk_builder
    → embedding (OpenRouter)
    → PostgreSQL (metadata, filter)
    → OpenSearch (chunk_text + knn embedding + filters)
```

Search cho agent:

| Tool | Kiểu |
|------|------|
| Semantic | kNN trên `embedding` |
| Keyword | multi_match text |
| Filter | term filters (category, mệnh, màu, …) |
| Image | SigLIP vector → kNN |

---

## Dataset fine-tune phong thủy (tham khảo)

`dataset_fengshui_cot/` — chat JSONL + Chain-of-Thought:

- Task **menh**: năm sinh / can chi → nạp âm, ngũ hành, màu hợp–kỵ  
- Task **size**: cổ tay → size li, số hạt, cung Sinh–Lão–Bệnh–Tử  
- Split train / valid / test (+ `test_extrapolate` năm OOD)

Logic này được dùng trong chatbot (tool FT hoặc code fallback).

---

## Lưu ý

- **Không commit `.env`** (đã gitignore).  
- Giá / tồn / KM có thể cập nhật bằng admin Excel mà không cần chạy lại full pipeline.  
- Tắt finetune: để trống `FINETUNE_API_URL` / `FENGSHUI_API_URL` rồi restart chatbot.  
- Log chatbot: volume `chatbot_logs` hoặc `logs/` khi chạy local.

---

## License / mục đích

Đồ án / hệ thống tư vấn sản phẩm phong thủy nội bộ shop **Vạn An Group**.
