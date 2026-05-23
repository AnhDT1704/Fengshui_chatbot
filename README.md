# Fengshui Products – Data Processing Pipeline

Xử lý dữ liệu sản phẩm phong thủy cho hệ thống Agentic RAG Chatbot.

## Kiến trúc

```
Raw .txt files
     │
     ▼
┌─────────────┐     ┌──────────────────┐
│  Parser      │────▶│  Metadata        │
│  (split by   │     │  Extractor       │
│   --N--)     │     │  (regex-based)   │
└─────────────┘     └──────┬───────────┘
                           │
                    ┌──────▼───────────┐
                    │  Chunk Builder    │
                    │  (remove         │
                    │   boilerplate)   │
                    └──────┬───────────┘
                           │
              ┌────────────┼────────────┐
              ▼                         ▼
   ┌──────────────────┐     ┌───────────────────┐
   │   PostgreSQL      │     │   OpenSearch       │
   │   (metadata,      │     │   (chunk_text,     │
   │    filter search) │     │    embedding,      │
   │                   │     │    metadata filter) │
   └──────────────────┘     └───────────────────┘
```

## Cấu trúc file

```
fengshui_data_pipeline/
├── docker-compose.yml        # PostgreSQL + OpenSearch + Dashboards
├── requirements.txt
├── .env.example              # Copy thành .env, điền API key
├── config.py                 # Load config từ .env
├── product_parser.py         # Parse .txt → raw product list
├── metadata_extractor.py     # Extract metadata (category, material, mệnh...)
├── chunk_builder.py          # Build enriched chunks, loại boilerplate
├── embedding_service.py      # Gọi OpenRouter text-embedding-3-small
├── models.py                 # SQLAlchemy models (Product, Boilerplate)
├── db_service.py             # PostgreSQL CRUD
├── opensearch_service.py     # OpenSearch index + CRUD + search
├── pipeline.py               # Main pipeline (CLI)
└── data/                     # Đặt file .txt ở đây
    ├── 40_san_pham_numbered.txt
    └── --41--.txt
```

## Setup

### 1. Cài Docker services

```bash
docker compose up -d
```

Chờ khoảng 30s cho OpenSearch khởi động. Kiểm tra:

```bash
curl http://localhost:9200        # OpenSearch
curl http://localhost:5432        # PostgreSQL (dùng psql)
curl http://localhost:8000/health # Chatbot API
```

`docker compose up -d` chạy 5 service: `postgres`, `pgadmin`, `opensearch`,
`opensearch-dashboards`, và `chatbot` (FastAPI). Chatbot mount `./` vào `/app`
nên sửa code trên host sẽ tự reload trong container.

```bash
docker compose logs -f chatbot       # xem log realtime
docker compose restart chatbot       # restart nhanh
docker compose build chatbot         # rebuild khi đổi requirements.txt
docker compose up -d postgres opensearch   # chỉ chạy hạ tầng, không chạy chatbot
```

Trong container, `PG_HOST=postgres` và `OS_HOST=opensearch` được override (giá trị
`localhost` trong `.env` chỉ dùng khi chạy bằng venv ở host).

### 2. Cài Python dependencies

```bash
python -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 3. Cấu hình

```bash
cp .env.example .env
# Sửa OPENROUTER_API_KEY trong .env
```

### 4. Đặt data files

```bash
mkdir -p data
cp /path/to/40_san_pham_numbered.txt data/
cp /path/to/--41--.txt data/
```

## Sử dụng

### Chạy toàn bộ pipeline

```bash
python pipeline.py
```

### Chạy từng bước

```bash
# Chỉ parse và extract metadata (không cần API key)
python pipeline.py --steps 1,2,3 --save-json

# Kiểm tra output
cat output/processed_products.json | python -m json.tool | head -50

# Chạy embedding + indexing
python pipeline.py --steps 4,5,6,7
```

### Reset database

```bash
python pipeline.py --reset-db
```

### Test riêng từng module

```bash
python product_parser.py         # Test parser
python metadata_extractor.py     # Test metadata extraction
python chunk_builder.py          # Test chunk building
python embedding_service.py      # Test embedding API
```

## Schema PostgreSQL

| Column                | Type          | Mô tả                                  |
|-----------------------|---------------|-----------------------------------------|
| `product_id`          | INTEGER       | ID sản phẩm (--N--)                    |
| `name`                | VARCHAR(500)  | Tên sản phẩm                           |
| `category`            | VARCHAR(100)  | vòng tay, nhang, treo xe...            |
| `material`            | TEXT[]        | ["aquamarine"], ["trầm hương"]         |
| `compatible_elements` | TEXT[]        | ["Thủy", "Mộc"]                       |
| `colors`              | TEXT[]        | ["xanh dương", "trắng"]               |
| `bead_sizes`          | TEXT[]        | ["6mm", "8mm"]                         |
| `price`               | FLOAT         | Giá (cần bổ sung thủ công)            |
| `in_stock`            | BOOLEAN       | Còn hàng (default: true)              |
| `raw_text`            | TEXT          | Nội dung gốc                           |
| `chunk_text`          | TEXT          | Chunk đã enriched                      |

## Schema OpenSearch

Mỗi document trong index `fengshui_products` gồm:

- `embedding` (knn_vector, 1536 dims) → Semantic Search
- `chunk_text` (text, vi_analyzer) → Keyword Search
- `name`, `category`, `material`, `compatible_elements`, `colors`... (keyword) → Filter Search

## Search Tools cho Agent

| Tool              | Query type                    | OpenSearch API              |
|-------------------|-------------------------------|-----------------------------|
| Semantic Search   | "đá hợp mệnh Thủy"          | kNN trên field `embedding`  |
| Keyword Search    | "aquamarine", "nhang trầm"   | multi_match trên text fields|
| Filter Search     | category=nhang, mệnh=Thủy    | bool query + term filters   |

## Lưu ý

- **Giá sản phẩm**: Dữ liệu gốc không có giá → cần bổ sung thủ công qua admin dashboard hoặc SQL UPDATE.
- **Mệnh phong thủy**: Một số sản phẩm không ghi rõ mệnh → `compatible_elements` sẽ trống, cần review thủ công.
- **Boilerplate**: Phần bảo quản và cam kết được loại khỏi chunk nhưng vẫn lưu trong `raw_text` tại PostgreSQL.
