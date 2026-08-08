"""
knowledge_base_agent.py – Tool-using agent for everything related to product data.

Tools (LLM picks based on docstring):
  - semantic_search_tool : natural-language descriptive query
  - keyword_search_tool : a specific stone / material / proper noun
  - filter_search_tool : structured filters (category, material, color, element)
  - get_product_detail_tool: deep-dive on one product (by id)
  - product_care_tool : usage / care guidelines
  - fengshui_advisor_tool : birth_year → Can Chi → Nạp âm → mệnh + màu hợp (theo
                             quy luật tương sinh) + ví dụ đá CÓ THẬT trong kho shop
                             (code 60-year cycle, HOẶC model finetune nếu
                             FENGSHUI_API_URL), then chain into filter_search
  - image_search_tool : VISUAL SEARCH — embed ảnh khách (SigLIP 2) → kNN trên
                             index ảnh OpenSearch → nhận diện đúng sản phẩm (ngưỡng)
  - analyze_image_tool : (phụ) mô tả ảnh → embed text → semantic search
  - get_product_images_tool: lấy URL ảnh của 1 sản phẩm theo id

Gemini là model multimodal — khi user gửi ảnh, ảnh nằm trong HumanMessage và LLM
"nhìn" được trực tiếp, nên KB agent xử lý luôn câu hỏi kèm ảnh (không cần agent riêng).

All tools enrich OpenSearch hits with PostgreSQL rows so the LLM sees the full
product (price_range, quantity_max, image URL, full description).
"""

from __future__ import annotations

import _bootstrap # noqa: F401

import base64
import contextvars
import json
import os
import re
import time
from pathlib import Path
from typing import Annotated, Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

import requests

import db_service
import fengshui_finetune_client as fengshui_ft
import image_embedding
import opensearch_service
import progress
from embedding_service import embed_single
from gemini import make_llm, make_llm_with_tools
from logger import ToolLoggerCallback, get_logger


# Ảnh khách gửi (bytes) của lượt hiện tại — set trong run(), đọc trong
# image_search_tool. Dùng contextvar để an toàn khi nhiều request song song.
_QUERY_IMAGE: contextvars.ContextVar = contextvars.ContextVar("kb_query_image", default=None) # list[bytes]

# Ngưỡng cosine để coi ảnh khách là "đúng sản phẩm shop" (đã hiệu chỉnh từ POC:
# match đúng ~0.90-0.96, sản phẩm khác ≤0.79).
IMAGE_MATCH_THRESHOLD = float(os.getenv("IMAGE_MATCH_THRESHOLD", "0.85"))

# Đặt trong .env: FINETUNE_API_URL=https://xxxx.ngrok-free.app
# Bật → ảnh khách → model finetune nhận diện (/predict); tắt SigLIP;
# DB chỉ tra giá/tồn theo TÊN model nhận ra. Trống → chatbot chạy y như cũ.
# Khác FENGSHUI_API_URL (model phong thủy text — xem fengshui_finetune_client).
FINETUNE_API_URL = os.getenv("FINETUNE_API_URL", "").rstrip("/")
USE_FINETUNE = bool(FINETUNE_API_URL)
_SEED_TOOL_NAME = "keyword_search_tool" if USE_FINETUNE else "image_search_tool"

# full=True : model sinh ĐỦ 7 cột, gồm cả product_description (~90-140s/ảnh).
# full=False : DỪNG SỚM ngay trước product_description (~8s/ảnh) ← mặc định.
# Vì sao tắt được mà KHÔNG mất gì: description model sinh ra vốn đã bị VỨT ĐI. Nhìn
# finetune_identify() bên dưới — sản phẩm tiêm vào hội thoại là `prod` lấy từ POSTGRES
# (_enrich_with_pg → _serialize_product, đã kèm product_description của DB); thứ model
# đoán chỉ nằm trong `_finetune_attrs` để tham chiếu. Nên bật full chỉ tổ bắt khách chờ
# thêm ~80 giây để sinh một đoạn văn rồi ném đi.
# Cắt được là nhờ product_description là trường CUỐI trong JSON — dừng sinh chữ trước
# nó thì 6 trường kia đã xong.
_FINETUNE_FULL: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "finetune_full", default=False
)


def set_finetune_full(value: bool):
    """Đặt chế độ cho request hiện tại (api.py gọi trước khi chạy graph)."""
    return _FINETUNE_FULL.set(bool(value))


def _extract_query_images_bytes(messages) -> list[bytes]:
    """Lấy bytes của TẤT CẢ ảnh trong HumanMessage mới nhất (data-URI base64 hoặc http URL)."""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            content = m.content
            out: list[bytes] = []
            if isinstance(content, list):
                for part in content:
                    if not (isinstance(part, dict) and part.get("type") == "image_url"):
                        continue
                    url = (part.get("image_url") or {}).get("url", "")
                    data: Optional[bytes] = None
                    if url.startswith("data:"):
                        try:
                            data = base64.b64decode(url.split(",", 1)[1])
                        except Exception:
                            data = None
                    elif url.startswith("http"):
                        try:
                            data = image_embedding.download_bytes(url)
                        except Exception:
                            data = None
                    if data:
                        out.append(data)
            return out # chỉ xét lượt người dùng mới nhất
    return []


def _latest_image_data_urls(messages) -> list[str]:
    """Lấy URL ảnh từ HumanMessage GẦN NHẤT CÓ ẢNH.

    QUAN TRỌNG: bỏ qua các HumanMessage text-only chèn SAU ảnh (vd '[GHI CHÚ NỘI BỘ]'
    mà chuỗi agent skills→KB tự thêm). Nếu chỉ xét message mới nhất, ảnh của khách bị
    che → KB tưởng không có ảnh → bỏ qua nhận diện VLM (bug ảnh+size)."""
    for m in reversed(messages):
        if isinstance(m, HumanMessage) and isinstance(m.content, list):
            urls = [
                (part.get("image_url") or {}).get("url", "")
                for part in m.content
                if isinstance(part, dict) and part.get("type") == "image_url"
            ]
            urls = [u for u in urls if u]
            if urls:
                return urls
    return []


def _data_url_to_bytes(url: str) -> Optional[bytes]:
    if url.startswith("data:") and "," in url:
        try:
            return base64.b64decode(url.split(",", 1)[1])
        except Exception:
            return None
    if url.startswith("http"):
        try:
            return image_embedding.download_bytes(url)
        except Exception:
            return None
    return None


_IDENTIFY_PROMPT = (
    "Bạn xem các ảnh khách gửi cho shop phong thủy. Với TỪNG ảnh theo đúng thứ tự, trả về:\n"
    "- name: tên sản phẩm IN trên ảnh (dòng chữ to trong ảnh bao bì/quảng cáo), hoặc "
    "null nếu ảnh KHÔNG có chữ tên.\n"
    "- is_product: true nếu ảnh là SẢN PHẨM PHONG THỦY / TRANG SỨC / VẬT PHẨM loại shop "
    "bán (vòng tay, chuỗi hạt, đá phong thủy, nhang, trầm hương, lư xông trầm, thác khói, "
    "tượng phật, dây treo xe, dây chuyền, mặt dây...); false nếu là thứ KHÁC không liên "
    "quan (người, thú cưng, xe cộ, đồ ăn, phong cảnh, ảnh chụp màn hình/chữ, vật dụng "
    "thông thường...).\n"
    'CHỈ trả về JSON: {"items": [{"name": "<tên hoặc null>", "is_product": true/false}, ...]}. '
    "Không thêm bất kỳ chữ nào ngoài JSON."
)


def _fuse_pick(
    name_set: set[int],
    name_pids: list[int],
    visual: list[tuple[int, float]],
) -> Optional[int]:
    """Chọn 1 product_id từ 2 nguồn: tên (name_pids, đã khoanh dòng) + visual
    (đã xếp theo cosine giảm dần). Trả None nếu không có gì."""
    # 1) Hit visual cao nhất mà cũng nằm trong nhóm khớp tên → đúng cả dòng lẫn biến thể.
    for pid, _cos in visual:
        if pid in name_set:
            return pid
    # 2) Hai nguồn không giao nhau → tin visual nếu cosine đủ cao (nhận diện ảnh chắc chắn).
    if visual and visual[0][1] >= IMAGE_MATCH_THRESHOLD:
        return visual[0][0]
    # 3) Cuối cùng: tin tên (nếu đọc được), rồi mới tới visual top-1.
    if name_pids:
        return name_pids[0]
    if visual:
        return visual[0][0]
    return None


def identify_image_products(messages) -> dict:
    """Nhận diện sản phẩm THẬT từ ảnh khách gửi BẰNG CODE (không phụ thuộc model
    chịu gọi tool hay không):
      0) phân loại ảnh có phải SẢN PHẨM loại shop bán không (is_product) — lọc bỏ
         ảnh không liên quan (người/thú/xe/đồ ăn...),
      1) đọc tên in trên từng ảnh bằng 1 lượt vision,
      2) keyword_search tên đó trong DB (khoanh dòng),
      3) vector ảnh SigLIP (chọn biến thể/màu); fusion 2 nguồn.

    Trả về dict:
      - has_image: lượt này có ảnh không
      - any_product_like: có ÍT NHẤT 1 ảnh là sản phẩm loại shop bán
      - products: list sản phẩm thật (đã _serialize_product), không trùng id
    """
    urls = _latest_image_data_urls(messages)
    if not urls:
        return {"has_image": False, "any_product_like": False, "products": []}

    # 1) Vision call: đọc tên + phân loại is_product cho từng ảnh (1 lượt).
    items: list = []
    try:
        vis = make_llm(temperature=0.0, max_tokens=1024)
        content = [{"type": "text", "text": _IDENTIFY_PROMPT}]
        for u in urls:
            content.append({"type": "image_url", "image_url": {"url": u}})
        resp = vis.invoke([HumanMessage(content=content)])
        raw = resp.content if isinstance(resp.content, str) else str(resp.content)
        s, e = raw.find("{"), raw.rfind("}")
        if s != -1 and e != -1:
            items = (json.loads(raw[s:e + 1]) or {}).get("items", []) or []
    except Exception as ex:
        log.warning("identify: vision đọc ảnh lỗi: %s", ex)

    products: list[dict] = []
    seen: set[int] = set()
    any_product_like = False
    for i, url in enumerate(urls):
        item = items[i] if (i < len(items) and isinstance(items[i], dict)) else {}
        nm = item.get("name")
        # Mặc định is_product=True khi KHÔNG có tín hiệu rõ (vision lỗi/thiếu) → tránh
        # từ chối nhầm ảnh sản phẩm thật.
        is_prod = bool(item.get("is_product", True))

        # (b) VECTOR ẢNH (SigLIP) — luôn chạy, dùng để chọn biến thể + làm "lưới an toàn"
        # cho relevance (cosine cao thì gần như chắc là sản phẩm shop).
        visual: list[tuple[int, float]] = []
        try:
            b = _data_url_to_bytes(url)
            if b:
                vec = image_embedding.embed_image(b)
                vhits = opensearch_service.image_knn_search(vec.tolist(), k=20)
                best: dict[int, float] = {}
                for h in vhits:
                    pid, cos = h["product_id"], h["cosine"]
                    if pid not in best or cos > best[pid]:
                        best[pid] = cos
                visual = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        except Exception as ex:
            log.warning("identify: visual embed lỗi: %s", ex)
        top_cos = visual[0][1] if visual else 0.0

        # (0) RELEVANCE: ảnh liên quan shop nếu vision bảo là sản phẩm, HOẶC ảnh khớp
        # rất cao với 1 sản phẩm trong index (vision có thể phân loại sót).
        relevant = is_prod or top_cos >= IMAGE_MATCH_THRESHOLD
        if not relevant:
            log.info("identify: ảnh #%d KHÔNG liên quan sản phẩm shop (is_product=False, cos=%.3f)",
                     i + 1, top_cos)
            continue
        any_product_like = True

        # (a) TÊN → khoanh đúng DÒNG sản phẩm (nhiều biến thể cùng tên).
        name_pids: list[int] = []
        if isinstance(nm, str) and nm.strip():
            try:
                khits = opensearch_service.keyword_search(nm.strip(), k=10)
                name_pids = [h["product_id"] for h in khits if h.get("product_id") is not None]
            except Exception as ex:
                log.warning("identify: keyword_search('%s') lỗi: %s", nm, ex)

        # (c) FUSION: visual cosine cao nhất MÀ cũng khớp tên → đúng dòng lẫn biến thể.
        pid = _fuse_pick(set(name_pids), name_pids, visual)
        prod = None
        if pid is not None:
            enriched = _enrich_with_pg([{"product_id": pid}])
            prod = enriched[0] if enriched else None

        if prod and prod["product_id"] not in seen:
            seen.add(prod["product_id"])
            products.append(prod)

    log.info("identify_image_products → %d ảnh, any_product=%s, %d sản phẩm: %s",
             len(urls), any_product_like, len(products), [p.get("name") for p in products])
    return {"has_image": True, "any_product_like": any_product_like, "products": products}


def finetune_identify(messages) -> dict:
    """Nhận diện sản phẩm từ ảnh khách BẰNG MODEL FINETUNE (qua FINETUNE_API_URL).

    Model trả {name, category, colors, product_size, compatible_elements}. Ta dùng
    `name` tra DB (keyword_search) để lấy TÀI SẢN thật: product_id, ảnh, giá, tồn kho
    — vì model cố ý KHÔNG nhớ URL ảnh/giá (dữ liệu biến động). Trả về cùng cấu trúc
    với identify_image_products() để _seed_messages() dùng chung.
    """
    urls = _latest_image_data_urls(messages)
    if not urls:
        return {"has_image": False, "any_product_like": False, "products": []}

    products: list[dict] = []
    seen: set[int] = set()
    api_error: Optional[str] = None # API finetune sập (ngrok chết / timeout / 500)
    full = _FINETUNE_FULL.get() # nút gạt trên UI (mặc định False = nhanh)
    per_image: list[float] = [] # thời gian model xử lý TỪNG ảnh (giây)
    for i, url in enumerate(urls):
        b = _data_url_to_bytes(url)
        if not b:
            continue
        # Bước TỐN THỜI GIAN NHẤT. Báo cho khách biết để họ không tưởng bot treo.
        wait_hint = "khoảng 1-2 phút" if full else "vài giây"
        progress.emit(
            "identifying",
            (f"Shop đang xác minh sản phẩm trong ảnh {i + 1}/{len(urls)}, "
             f"bạn chờ {wait_hint} nhé...") if len(urls) > 1 else
            f"Shop đang xác minh sản phẩm trong ảnh, bạn chờ {wait_hint} nhé...",
            image_index=i + 1, total_images=len(urls), full=full,
        )
        # 1) Gọi MODEL FINETUNE trên Colab
        mode = "ĐẦY ĐỦ (có product_description)" if full else "NHANH (bỏ product_description)"
        t0 = time.perf_counter()
        try:
            resp = requests.post(
                f"{FINETUNE_API_URL}/predict",
                files={"file": ("image.jpg", b, "image/jpeg")},
                # full=false → server DỪNG SỚM trước product_description. Đo thực tế trên
                # T4: ~41s so với ~92s → nhanh hơn ~2.2 lần. Mô tả vẫn có, lấy từ Postgres.
                data={"full": "true" if full else "false"},
                headers={"ngrok-skip-browser-warning": "true"}, # tránh trang cảnh báo ngrok free
                # Qwen-7B 4-bit trên T4: chế độ đầy đủ đo được ~90-140s/ảnh (dao động mạnh
                # vì T4 free bị chia sẻ). Để 5 phút cho chắc.
                timeout=float(os.getenv("FINETUNE_TIMEOUT", "300")),
            )
            resp.raise_for_status()
            attrs = (resp.json() or {}).get("result", {}) or {}
            dt = time.perf_counter() - t0
            per_image.append(dt)
            # Dòng TIMING riêng — grep: TIMING.*FINETUNE_IMAGE
            log.info(
                "[TIMING] FINETUNE_IMAGE | status=ok | image=%d/%d | latency_s=%.3f | mode=%s | url=%s | name=%s",
                i + 1, len(urls), dt, "full" if full else "fast",
                FINETUNE_API_URL,
                (attrs.get("name") or "")[:80],
            )
            # Log ĐO THỜI GIAN + JSON NGUYÊN VĂN model trả về → vừa theo dõi được model
            # chậm/nhanh ở chế độ nào, vừa soi được model "nhìn" ra gì để đối chiếu với
            # Postgres ở dưới khi nghi nó đoán sai.
            log.info(
                "MODEL FINETUNE ẢNH ảnh #%d/%d %.1f giây chế độ %s\n%s\n",
                i + 1, len(urls), dt, mode,
                json.dumps(attrs, ensure_ascii=False, indent=2),
            )
        except Exception as ex:
            dt = time.perf_counter() - t0
            per_image.append(dt)
            log.warning(
                "[TIMING] FINETUNE_IMAGE | status=error | image=%d/%d | latency_s=%.3f | mode=%s | url=%s | error=%s",
                i + 1, len(urls), dt, "full" if full else "fast",
                FINETUNE_API_URL, ex,
            )
            log.warning("MODEL FINETUNE ẢNH: ảnh #%d THẤT BẠI sau %.1f giây (chế độ %s)",
                        i + 1, dt, mode)
            # API SẬP (ngrok chết / Colab ngắt / timeout / 500) — KHÁC HẲN với "ảnh không
            # phải sản phẩm shop". Ghi cờ để run() báo lỗi kỹ thuật, TUYỆT ĐỐI không nói
            # với khách là ảnh của họ không phải sản phẩm của shop (sai sự thật).
            api_error = str(ex)
            log.warning("finetune_identify: gọi API lỗi (ảnh #%d): %s", i + 1, ex)
            continue

        name = attrs.get("name")
        if not name or not str(name).strip():
            continue

        # Model đã đọc ra tên → khoe ngay cho khách (bằng chứng hệ thống chạy đúng),
        # rồi mới tra DB lấy giá/tồn kho.
        progress.emit(
            "identified",
            f"Đã nhận ra: {str(name).strip()}\nĐang tra giá và tồn kho...",
            product_name=str(name).strip(),
        )

        # 2) Map name -> sản phẩm THẬT trong DB để lấy ảnh/giá/tồn kho
        prod = None
        try:
            hits = opensearch_service.keyword_search(str(name).strip(), k=1)
            if hits:
                enriched = _enrich_with_pg([{"product_id": hits[0]["product_id"]}])
                prod = enriched[0] if enriched else None
        except Exception as ex:
            log.warning("finetune_identify: keyword_search('%s') lỗi: %s", name, ex)

        if prod:
            prod["_finetune_attrs"] = attrs # thuộc tính model đoán (để tham chiếu)
            # Đối chiếu: model ĐOÁN gì vs Postgres CÓ gì. Giá/tồn kho KHÔNG train nên
            # chỉ DB mới có — đây là chỗ thấy rõ ranh giới đó.
            log.info(
                "MAP vào DB: product_id=%s | %s\n"
                "model đoán : category=%s material=%s colors=%s size=%s\n"
                "Postgres : category=%s material=%s colors=%s size=%s\n"
                "CHỈ CÓ Ở DB: giá=%s | tồn=%s | còn hàng=%s",
                prod["product_id"], prod["name"][:55],
                attrs.get("category"), attrs.get("material"),
                attrs.get("colors"), attrs.get("product_size"),
                prod.get("category"), prod.get("material"),
                prod.get("colors"), prod.get("product_size"),
                prod.get("price_range"), prod.get("quantity_max"), prod.get("in_stock"),
            )
            if prod["product_id"] not in seen:
                seen.add(prod["product_id"])
                products.append(prod)
        else:
            # Không map được vào DB → vẫn trả thuộc tính model đoán (thiếu ảnh/giá)
            products.append({
                "product_id": None,
                "name": name,
                "category": attrs.get("category"),
                "material": attrs.get("material", []),
                "colors": attrs.get("colors", []),
                "product_size": attrs.get("product_size", []),
                "compatible_elements": attrs.get("compatible_elements", []),
                "image_cover": None,
                "_finetune_attrs": attrs,
            })

    # Tổng kết thời gian: dòng này để theo dõi model chậm dần hay Colab bị bóp GPU.
    if per_image:
        total_s = sum(per_image)
        avg_s = total_s / len(per_image)
        log.info(
            "[TIMING] FINETUNE_IMAGE | status=summary | images=%d | total_s=%.3f | avg_s=%.3f | mode=%s | url=%s",
            len(per_image), total_s, avg_s, "full" if full else "fast", FINETUNE_API_URL,
        )
        log.info(
            "MODEL FINETUNE ẢNH tổng %.1f giây / %d ảnh (trung bình %.1f s/ảnh) — chế độ %s",
            total_s, len(per_image), avg_s,
            "ĐẦY ĐỦ" if full else "NHANH",
        )

    log.info("finetune_identify → %d ảnh, %d sản phẩm: %s (api_error=%s)",
             len(urls), len(products), [p.get("name") for p in products], api_error)
    return {
        "has_image": True,
        "any_product_like": bool(products),
        "products": products,
        # Chỉ coi là sự cố khi API lỗi VÀ không nhận được sản phẩm nào.
        "api_error": api_error if not products else None,
    }


log = get_logger("kb")
_callback = ToolLoggerCallback("kb")

_USAGE_GUIDELINES_PATH = Path(__file__).parent / "usage_guidelines.json"
_USAGE_GUIDELINES = json.loads(_USAGE_GUIDELINES_PATH.read_text(encoding="utf-8"))


def _serialize_product(product) -> dict:
    """Serialize a SQLAlchemy Product row to a JSON-friendly dict."""
    image_cover = None
    if product.image:
        if isinstance(product.image, list) and product.image:
            image_cover = product.image[0]
        elif isinstance(product.image, dict):
            image_cover = product.image.get("cover") or next(iter(product.image.values()), None)

    return {
        "product_id": product.product_id,
        "name": product.name,
        "category": product.category,
        "material": list(product.material or []),
        "compatible_elements": list(product.compatible_elements or []),
        "colors": list(product.colors or []),
        "product_size": list(product.product_size or []),
        # Cột THỜI GIAN THỰC — cố ý KHÔNG train vào model finetune (giá/tồn kho đổi
        # liên tục). Luôn lấy tươi từ Postgres theo product_id mà finetune nhận ra.
        "price_range": product.price_range,
        "in_stock": bool(product.in_stock),
        "quantity_min": getattr(product, "quantity_min", None),
        "quantity_max": getattr(product, "quantity_max", None),
        "image_cover": image_cover,
        "product_description": product.product_description,
        # bảo hành RIÊNG của SP (cột DB, không phải VLM) — vd "thay dây trọn đời", "24 tháng"
        "warranty": getattr(product, "warranty", None),
    }


def _enrich_with_pg(hits: list[dict]) -> list[dict]:
    """Given OpenSearch hits, fetch PG rows and return merged product objects."""
    enriched = []
    for hit in hits:
        pid = hit.get("product_id")
        if pid is None:
            continue
        product = db_service.get_product_by_id(pid)
        if product is None:
            continue
        merged = _serialize_product(product)
        if "score" in hit:
            merged["_score"] = hit["score"]
        enriched.append(merged)
    return enriched


def _format_for_llm(products: list[dict]) -> str:
    """Compact JSON suitable to stuff into the LLM context."""
    return json.dumps(products, ensure_ascii=False, indent=2)


CAN = ["Giáp","Ất","Bính","Đinh","Mậu","Kỷ","Canh","Tân","Nhâm","Quý"]
CHI = ["Tý","Sửu","Dần","Mão","Thìn","Tỵ","Ngọ","Mùi","Thân","Dậu","Tuất","Hợi"]

# 30 Nạp âm — each covers 2 consecutive years
NAPAM: list[tuple[str, str]] = [
    ("Hải Trung Kim", "Kim"), ("Lư Trung Hỏa", "Hỏa"),
    ("Đại Lâm Mộc", "Mộc"), ("Lộ Bàng Thổ", "Thổ"),
    ("Kiếm Phong Kim","Kim"), ("Sơn Đầu Hỏa", "Hỏa"),
    ("Giản Hạ Thủy", "Thủy"), ("Thành Đầu Thổ","Thổ"),
    ("Bạch Lạp Kim", "Kim"), ("Dương Liễu Mộc","Mộc"),
    ("Tỉnh Tuyền Thủy","Thủy"), ("Ốc Thượng Thổ","Thổ"),
    ("Tích Lịch Hỏa","Hỏa"), ("Tùng Bách Mộc","Mộc"),
    ("Trường Lưu Thủy","Thủy"), ("Sa Trung Kim", "Kim"),
    ("Sơn Hạ Hỏa", "Hỏa"), ("Bình Địa Mộc", "Mộc"),
    ("Bích Thượng Thổ","Thổ"), ("Kim Bạc Kim", "Kim"),
    ("Phú Đăng Hỏa", "Hỏa"), ("Thiên Hà Thủy", "Thủy"),
    ("Đại Trạch Thổ","Thổ"), ("Thoa Xuyến Kim","Kim"),
    ("Tang Đố Mộc", "Mộc"), ("Đại Khê Thủy", "Thủy"),
    ("Sa Trung Thổ", "Thổ"), ("Thiên Thượng Hỏa","Hỏa"),
    ("Thạch Lựu Mộc","Mộc"), ("Đại Hải Thủy", "Thủy"),
]

# Trong phong thủy, MÀU của đá quyết định hành (không phải tên đá). Một viên đá
# mang hành E hợp người mệnh E (bản mệnh) + người mệnh mà E SINH RA (tương sinh).
# Vòng tương sinh: Mộc→Hỏa→Thổ→Kim→Thủy→Mộc.
# Mỗi mệnh có "lucky_color_groups" xếp theo ƯU TIÊN:
# group[0] = màu TƯƠNG SINH (đại cát, ưu tiên 1)
# group[1] = màu BẢN MỆNH (ưu tiên 2)
# "example_stones" CHỈ liệt kê đá shop THỰC SỰ bán đúng nhóm màu đó (đối chiếu
# cột material trong DB) — không bịa tên đá ngoài kho.
# Token màu khớp với giá trị cột "colors" trong DB để chain filter_search được.
# Quy ước key (tiếng Anh để LLM reasoning nhất quán; giá trị giữ tiếng Việt vì là
# nội dung domain / hiển thị cho khách):
# generating_element = hành SINH ra mệnh này (tương sinh, đại cát)
# controlling_element = hành KHẮC mệnh này (tương khắc, đại kỵ)

# Đá đa sắc cân bằng cả ngũ hành → hợp MỌI mệnh (shop có thật):
MULTICOLOR_STONES_ALL_ELEMENTS = ["mã não đa sắc", "tourmaline", "vòng ngũ sắc"]

ELEMENT_INFO = {
    "Kim": {
        "generating_element": "Thổ", # Thổ sinh Kim (đại cát)
        "controlling_element": "Hỏa", # Hỏa khắc Kim (đại kỵ)
        "lucky_color_groups": [
            {"colors": ["vàng", "nâu"], "reason": "Thổ sinh Kim (tương sinh, ưu tiên) — kích tài lộc, vững chãi"},
            {"colors": ["trắng"], "reason": "bản mệnh Kim (trắng/trong suốt) — thuần khiết, tỉnh táo"},
        ],
        "unlucky_colors": ["đỏ", "hồng", "tím"],
        "example_stones": ["mắt mèo vàng", "trầm hương", "mã não trắng", "thạch anh"],
    },
    "Mộc": {
        "generating_element": "Thủy",
        "controlling_element": "Kim",
        "lucky_color_groups": [
            {"colors": ["đen", "xanh dương"], "reason": "Thủy sinh Mộc (tương sinh, ưu tiên) — nâng uy tín, mở tư duy, hút tài"},
            {"colors": ["xanh lá", "xanh rêu"], "reason": "bản mệnh Mộc — sinh sôi, giảm stress, sáng tạo"},
        ],
        "unlucky_colors": ["trắng"],
        "example_stones": ["mã não đen", "aquamarine", "mã não xanh lá", "mã não rêu", "mắt mèo xanh"],
    },
    "Thủy": {
        "generating_element": "Kim",
        "controlling_element": "Thổ",
        "lucky_color_groups": [
            {"colors": ["trắng"], "reason": "Kim sinh Thủy (tương sinh, ưu tiên) — khai thông trí tuệ, sáng suốt"},
            {"colors": ["đen", "xanh dương"], "reason": "bản mệnh Thủy — củng cố địa vị, hanh thông"},
        ],
        "unlucky_colors": ["vàng", "nâu"],
        "example_stones": ["mã não trắng", "thạch anh", "mã não đen", "aquamarine", "thạch anh xanh"],
    },
    "Hỏa": {
        "generating_element": "Mộc",
        "controlling_element": "Thủy",
        "lucky_color_groups": [
            {"colors": ["xanh lá", "xanh rêu"], "reason": "Mộc sinh Hỏa (tương sinh, ưu tiên) — điều hòa cảm xúc, mở quan hệ"},
            {"colors": ["đỏ", "hồng", "tím"], "reason": "bản mệnh Hỏa — nhiệt huyết, mạnh mẽ, quyết đoán"},
        ],
        "unlucky_colors": ["đen", "xanh dương"],
        "example_stones": ["mã não xanh lá", "mắt mèo xanh", "mắt mèo đỏ", "chỉ đỏ"],
    },
    "Thổ": {
        "generating_element": "Hỏa",
        "controlling_element": "Mộc",
        "lucky_color_groups": [
            {"colors": ["đỏ", "hồng", "tím"], "reason": "Hỏa sinh Thổ (tương sinh, ưu tiên) — tiếp năng lượng, thúc đẩy sự nghiệp"},
            {"colors": ["vàng", "nâu"], "reason": "bản mệnh Thổ — củng cố nội lực, hút tiền tài, ổn định"},
        ],
        "unlucky_colors": ["xanh lá", "xanh rêu"],
        "example_stones": ["mắt mèo đỏ", "chỉ đỏ", "mắt mèo vàng", "trầm hương"],
    },
}


def _year_to_can_chi(year: int) -> dict:
    """Map a birth year to Can-Chi, Nạp âm and Ngũ hành element."""
    offset = (year - 1924) % 60
    if offset < 0:
        offset += 60
    stem = CAN[offset % 10]
    chi = CHI[offset % 12]
    napam, element = NAPAM[offset // 2]
    return {
        "year": year,
        "can": stem,
        "chi": chi,
        "can_chi": f"{stem} {chi}",
        "napam": napam,
        "element": element,
    }


@tool
def semantic_search_tool(query: str, top_k: int = 10) -> str:
    """
    Tìm sản phẩm theo mô tả tự nhiên / ý nghĩa / công dụng.

    Dùng khi user mô tả sản phẩm bằng ngôn ngữ tự nhiên, không nêu tên đá/chất liệu
    cụ thể. Ví dụ: "đeo tay cho may mắn", "vòng nhẹ nhàng dịu mắt".

    Args:
        query: Câu truy vấn của user
        top_k: Số sản phẩm trả về (mặc định 10)
    """
    embedding = embed_single(query)
    hits = opensearch_service.semantic_search(embedding, k=top_k)
    products = _enrich_with_pg(hits)
    if not products:
        return json.dumps({
            "results": [],
            "huong_dan": "semantic_search KHÔNG có kết quả. Hãy thử keyword_search_tool "
                         "(từ khoá chính) hoặc filter_search_tool (category/màu). Nếu vẫn "
                         "trống → nói thẳng shop chưa có. TUYỆT ĐỐI KHÔNG bịa sản phẩm.",
        }, ensure_ascii=False)
    return _format_for_llm(products)


@tool
def keyword_search_tool(query: str, top_k: int = 10) -> str:
    """
    Tìm sản phẩm theo từ khoá cụ thể trong tên / mô tả (full-text, KHÔNG giới hạn
    danh sách từ khoá — tìm được bất kỳ từ nào xuất hiện trong tên/mô tả).

    Dùng khi user nhắc đích danh đá / chất liệu, ví dụ:
    "tourmaline", "aquamarine", "trầm hương", "thạch anh", "mã não" (đen/trắng/
    rêu/xanh lá/đa sắc), "mắt mèo", "beryl", "đồng", "gốm sứ", "chỉ đỏ", "vỏ quế".
    Với các LOẠI sản phẩm (lư, nhang, treo xe, chuỗi hạt, thác khói, tượng phật,
    dây chuyền) thì ưu tiên filter_search_tool(category=...) thay vì tool này.

    Args:
        query: Từ khoá tìm kiếm
        top_k: Số sản phẩm trả về
    """
    hits = opensearch_service.keyword_search(query, k=top_k)
    products = _enrich_with_pg(hits)
    if not products:
        # FALLBACK TỰ ĐỘNG (bằng code, không phụ thuộc LLM): keyword rỗng → chạy
        # semantic theo cùng query, trả về mẫu GẦN GIỐNG thay vì để trống.
        sem_hits = opensearch_service.semantic_search(embed_single(query), k=top_k)
        sem_products = _enrich_with_pg(sem_hits)
        if sem_products:
            return json.dumps({
                "keyword_empty_fallback_to_semantic": True,
                "note": "Shop không có đúng loại khách hỏi. CÁCH TRÌNH BÀY: nêu NGẮN GỌN "
                        "shop chưa có đúng loại đó, RỒI giới thiệu các sản phẩm dưới đây "
                        "bằng câu KHẲNG ĐỊNH, ví dụ 'nhưng shop có những sản phẩm này cho "
                        "bạn tham khảo:'. TUYỆT ĐỐI KHÔNG dùng từ 'gần giống' / 'tương tự' "
                        "/ 'na ná' — giới thiệu như sản phẩm CHÍNH THỨC của shop.",
                "results": json.loads(_format_for_llm(sem_products)),
            }, ensure_ascii=False)
        return json.dumps({
            "results": [],
            "huong_dan": "Cả keyword lẫn semantic đều trống — báo thẳng shop chưa có "
                         "loại này, gợi ý hỏi nhân viên hoặc web_search nếu shop không "
                         "bán. TUYỆT ĐỐI KHÔNG tự bịa ra sản phẩm/tên/giá nào.",
        }, ensure_ascii=False)
    return _format_for_llm(products)


@tool
def filter_search_tool(
    category: Optional[str] = None,
    material: Optional[str] = None,
    compatible_elements: Optional[str] = None,
    colors: Optional[str] = None,
    in_stock: Optional[bool] = None,
    price_range: Optional[str] = None,
    top_k: int = 10,
) -> str:
    """
    Lọc sản phẩm theo các thuộc tính có cấu trúc.

    Dùng khi user nêu rõ tiêu chí lọc: theo danh mục, chất liệu, mệnh phong thủy
    (Kim/Mộc/Thủy/Hỏa/Thổ), màu sắc. Có thể truyền nhiều tiêu chí cùng lúc.

    Args:
        category: Vd "vòng tay", "nhang", "lư xông trầm",...
        material: Vd "tourmaline", "mã não đen", "trầm hương"
        compatible_elements: Mệnh hợp - Kim | Mộc | Thủy | Hỏa | Thổ
        colors: Vd "đen", "xanh dương", "đa sắc"
        in_stock: True để chỉ lấy sản phẩm còn hàng
        price_range: Vd "100.000 - 200.000"
        top_k: Số sản phẩm trả về
    """
    filters = {}
    if category: filters["category"] = category
    if material: filters["material"] = material
    if compatible_elements: filters["compatible_elements"] = compatible_elements
    if colors: filters["colors"] = colors
    if in_stock is not None: filters["in_stock"] = in_stock
    if price_range: filters["price_range"] = price_range

    hits = opensearch_service.filter_search(filters, k=top_k)
    products = _enrich_with_pg(hits)
    if not products:
        return json.dumps({
            "results": [],
            "huong_dan": "filter_search KHÔNG có kết quả với bộ lọc này. Hãy NỚI tiêu "
                         "chí (bỏ bớt 1 filter) hoặc thử semantic_search_tool / "
                         "keyword_search_tool. Nếu mọi cách vẫn trống → nói thẳng shop "
                         "chưa có loại này. TUYỆT ĐỐI KHÔNG tự bịa ra sản phẩm/tên/giá.",
        }, ensure_ascii=False)
    return _format_for_llm(products)


@tool
def get_product_detail_tool(product_id: int) -> str:
    """
    Lấy đầy đủ thông tin một sản phẩm cụ thể theo product_id.

    Dùng khi user hỏi chi tiết về một sản phẩm đã được nhắc đến (vd: "cho tôi
    biết thêm về sản phẩm số 5", "vòng aquamarine kia bảo hành thế nào").

    QUAN TRỌNG: product_id PHẢI là id CÓ THẬT lấy từ kết quả search trong hội
    thoại này, hoặc số id khách đưa. KHÔNG ĐOÁN/BỊA id. Nếu chỉ biết TÊN sản phẩm
    → gọi keyword_search_tool(query=tên) lấy id trước rồi mới gọi tool này.

    Args:
        product_id: Mã sản phẩm (đã xác thực, không đoán)
    """
    product = db_service.get_product_by_id(product_id)
    if product is None:
        return json.dumps({
            "internal_error": "id_not_found",
            "instruction": (
                "LỖI NỘI BỘ (KHÔNG nói với khách, KHÔNG nhắc 'id'/'product_id'): id vừa truyền "
                "không khớp sản phẩm nào — có thể bạn đoán sai. HÃY tự gọi keyword_search_tool"
                "(query=TÊN sản phẩm đang nói) để lấy đúng sản phẩm rồi trả lời. Nếu sản phẩm đã "
                "hiển thị ở lượt trước, dùng luôn dữ liệu (gồm product_description) đã có trong "
                "hội thoại, KHÔNG cần gọi lại tool."
            ),
        }, ensure_ascii=False)
    return _format_for_llm([_serialize_product(product)])


@tool
def product_care_tool(product_id: Optional[int] = None) -> str:
    """
    Trả về TOÀN BỘ hướng dẫn SỬ DỤNG & BẢO QUẢN sản phẩm của shop.

    Dùng khi user hỏi: cách chỉnh vòng rộng/chật, đứt dây/thay dây, bảo quản
    trầm hương, đeo có đụng nước được không, cách đeo/điều chỉnh, cách đếm hạt...

    Tool KHÔNG tự lọc — nó đưa cả danh sách hướng dẫn. BẠN (agent) hãy TỰ ĐỌC và
    suy luận xem câu hỏi của khách thuộc tình huống nào trong danh sách rồi chọn
    thông tin (kèm link video nếu có) để trả lời. Mỗi mục có 'aliases' là vài
    cách khách hay nói, chỉ để bạn tham khảo khi đối chiếu ngữ nghĩa.

    Args:
        product_id: (tuỳ chọn) nếu khách hỏi bảo quản riêng 1 sản phẩm cụ thể,
                    truyền product_id để lấy thêm product_description của sản phẩm.
    """
    result = {
        "guidelines": _USAGE_GUIDELINES["guidelines"],
        "videos": _USAGE_GUIDELINES.get("videos", {}),
    }

    if product_id is not None:
        product = db_service.get_product_by_id(product_id)
        if product is not None:
            result["product_specific"] = {
                "product_id": product.product_id,
                "name": product.name,
                "product_description": product.product_description,
            }

    return json.dumps(result, ensure_ascii=False)


def _fengshui_result_from_code(birth_year: int) -> dict:
    """Nguồn sự thật code (chu kỳ 60 năm) — dùng mặc định hoặc fallback khi FT lỗi."""
    info = _year_to_can_chi(birth_year)
    element = info["element"]
    rel = ELEMENT_INFO[element]
    lucky_colors = [c for g in rel["lucky_color_groups"] for c in g["colors"]]
    generating = rel["generating_element"]
    controlling = rel["controlling_element"]
    return {
        **info,
        "personal_element": f"Mệnh {element}",
        "best_match_element": f"Mệnh {generating} (sinh ra {element}) — đại cát, mạnh nhất",
        "avoid_element": f"Mệnh {controlling} (khắc {element}) — nên tránh",
        "lucky_color_groups": rel["lucky_color_groups"],
        "lucky_colors": lucky_colors,
        "unlucky_colors": rel["unlucky_colors"],
        "example_stones": rel["example_stones"],
        "multicolor_stones": MULTICOLOR_STONES_ALL_ELEMENTS,
        "suggested_filter_elements": [element, generating],
        "explanation": (
            f"Bạn sinh năm {birth_year} - Can Chi {info['can_chi']} - "
            f"Nạp âm {info['napam']} - mệnh {element}. "
            f"Hợp nhất với sản phẩm thuộc mệnh {generating} (tương sinh) "
            f"và mệnh {element} (bản mệnh). Tránh mệnh {controlling}."
        ),
        "source": "code",
    }


def _element_result_from_code(element: str) -> dict:
    """Khi khách đã nêu mệnh (vd Thủy) — map ELEMENT_INFO (fallback, không FT)."""
    e = (element or "").replace("Mệnh", "").replace("mệnh", "").strip()
    # Chuẩn hoá viết hoa chữ cái đầu ngũ hành
    for cand in ELEMENT_INFO:
        if e.lower() == cand.lower() or e.lower() == f"mệnh {cand.lower()}":
            e = cand
            break
    if e not in ELEMENT_INFO:
        return {"error": f"Không nhận ra mệnh {element!r}. Hợp lệ: Kim/Mộc/Thủy/Hỏa/Thổ."}
    rel = ELEMENT_INFO[e]
    lucky = [c for g in rel["lucky_color_groups"] for c in g["colors"]]
    gen, ctrl = rel["generating_element"], rel["controlling_element"]
    return {
        "element": e,
        "personal_element": f"Mệnh {e}",
        "best_match_element": f"Mệnh {gen} (sinh ra {e}) — đại cát, mạnh nhất",
        "avoid_element": f"Mệnh {ctrl} (khắc {e}) — nên tránh",
        "lucky_color_groups": rel["lucky_color_groups"],
        "lucky_colors": lucky,
        "unlucky_colors": rel["unlucky_colors"],
        "example_stones": rel["example_stones"],
        "multicolor_stones": MULTICOLOR_STONES_ALL_ELEMENTS,
        "suggested_filter_elements": [e, gen],
        "explanation": (
            f"Mệnh {e}: hợp nhất sản phẩm mệnh {gen} (tương sinh) và mệnh {e} (bản mệnh). "
            f"Tránh mệnh {ctrl}."
        ),
        "source": "code",
    }


def _fengshui_result_from_ft(data: dict, birth_year: Optional[int] = None) -> dict:
    """Map JSON model finetune → shape tool (bổ sung example_stones / filter chain)."""
    if data.get("need_more_info"):
        out = {**data, "source": "fengshui_finetune", "ft_think": data.get("_think") or ""}
        return out

    element = (
        data.get("element")
        or (data.get("personal_element") or "").replace("Mệnh ", "").strip()
    )
    for cand in ELEMENT_INFO:
        if str(element).lower() == cand.lower():
            element = cand
            break

    by = birth_year if birth_year is not None else data.get("birth_year") or data.get("year")

    if element not in ELEMENT_INFO:
        # Model trả JSON lệch
        if by and 1900 <= int(by) <= 2100:
            log.warning("FENGSHUI FT menh thiếu/element lạ %r → fallback code year", element)
            out = _fengshui_result_from_code(int(by))
            out["source"] = "code_fallback"
            out["ft_raw"] = data
            out["ft_think"] = data.get("_think") or ""
            return out
        log.warning("FENGSHUI FT parse kém element=%r — trả raw + think cho agent", element)
        return {
            "source": "fengshui_finetune",
            "ft_think": data.get("_think") or "",
            "ft_model_json": data,
            "raw_model": data,
            "note": "Model chưa trả element chuẩn; hãy dựa ft_think/ft_model_json, "
                    "KHÔNG tự bịa quy luật ngũ hành ngoài dữ liệu tool.",
        }

    rel = ELEMENT_INFO[element]
    generating = data.get("generating_element") or rel["generating_element"]
    controlling = data.get("controlling_element") or rel["controlling_element"]
    lucky_colors = data.get("lucky_colors") or [
        c for g in rel["lucky_color_groups"] for c in g["colors"]
    ]
    unlucky = data.get("unlucky_colors") or rel["unlucky_colors"]
    can_chi = data.get("can_chi") or ""
    napam = data.get("napam") or data.get("nap_am") or ""

    explanation = data.get("explanation")
    if not explanation:
        parts = []
        if by:
            parts.append(f"Sinh năm {by}")
        if can_chi:
            parts.append(f"Can Chi {can_chi}")
        if napam:
            parts.append(f"Nạp âm {napam}")
        parts.append(f"mệnh {element}")
        explanation = (
            " - ".join(parts) + f". Hợp nhất SP mệnh {generating} (tương sinh) "
            f"và mệnh {element} (bản mệnh). Tránh mệnh {controlling}."
        )

    return {
        "year": by,
        "birth_year": by,
        "can_chi": can_chi,
        "napam": napam,
        "element": element,
        "personal_element": f"Mệnh {element}",
        "best_match_element": f"Mệnh {generating} (sinh ra {element}) — đại cát, mạnh nhất",
        "avoid_element": f"Mệnh {controlling} (khắc {element}) — nên tránh",
        "lucky_color_groups": rel["lucky_color_groups"],
        "lucky_colors": lucky_colors,
        "unlucky_colors": unlucky,
        "example_stones": rel["example_stones"],
        "multicolor_stones": MULTICOLOR_STONES_ALL_ELEMENTS,
        "suggested_filter_elements": [element, generating],
        "explanation": explanation,
        "source": "fengshui_finetune",
        "ft_latency_s": data.get("_latency_s"),
        "ft_think": data.get("_think") or "",
        "ft_model_json": {
            k: data[k] for k in (
                "task", "birth_year", "can_chi", "napam", "element",
                "generating_element", "controlling_element",
                "lucky_colors", "unlucky_colors", "need_more_info",
                "fengshui", "bead_count", "bead_size_li",
            ) if k in data
        },
        "query": data.get("_query") or "",
    }


def _log_fengshui_tool_result(result: dict, label: str = "menh") -> None:
    log.info(
        "CHATBOT dùng FENGSHUI (%s) source=%s\n"
        "query=%s birth_year=%s element=%s can_chi=%s\n"
        "lucky_colors=%s\n"
        "suggested_filter_elements=%s\n"
        "example_stones=%s\n"
        "ft_think (%d chars):\n%s\n"
        "tool payload keys=%s",
        label,
        result.get("source"),
        (result.get("query") or "")[:160],
        result.get("birth_year"),
        result.get("element"),
        result.get("can_chi"),
        result.get("lucky_colors"),
        result.get("suggested_filter_elements"),
        result.get("example_stones"),
        len(result.get("ft_think") or ""),
        (result.get("ft_think") or "(none)")[:4000],
        list(result.keys()),
    )


@tool
def fengshui_advisor_tool(
    query: str,
    birth_year: Optional[int] = None,
) -> str:
    """
   BẮT BUỘC gọi tool này cho MỌI câu hỏi liên quan PHONG THỦY trước khi trả lời:
      - mệnh (vd "tôi mệnh Thủy", "mệnh Hỏa đeo màu gì")
      - con giáp / tuổi (vd "tuổi Tý hợp đá nào")
      - năm sinh (vd "sinh 1990 hợp gì")
      - hợp/kỵ màu-đá, "có nên đeo vòng này theo mệnh không", nạp âm, can chi, ngũ hành

    Tool gọi MODEL FINETUNE phong thủy (hoặc code fallback). Agent CHỈ được dùng
    kết quả tool + dữ liệu SP trong DB/context — TUYỆT ĐỐI không tự suy ngũ hành
    bằng kiến thức Gemini.

    Args:
        query: nguyên câu hỏi / tóm tắt đủ ý khách (BẮT BUỘC). Vd "tôi mệnh Thủy
               có nên đeo vòng mã não đen không", "sinh năm 1990 mệnh gì".
        birth_year: năm sinh dương lịch nếu khách có cho (vd 1990). Có thì truyền thêm.
    """
    q = (query or "").strip()
    if not q and birth_year is None:
        return json.dumps({
            "error": "Cần query (câu hỏi phong thủy) hoặc birth_year.",
        }, ensure_ascii=False)

    if birth_year is not None and (birth_year < 1900 or birth_year > 2100):
        return json.dumps({
            "error": f"birth_year {birth_year} ngoài phạm vi hỗ trợ (1900-2100)",
        }, ensure_ascii=False)

    # Câu hỏi gửi model: ưu tiên query; bổ sung năm sinh nếu có
    if not q and birth_year is not None:
        q = f"Tôi sinh năm {birth_year}, mệnh gì vậy?"
    elif birth_year is not None and str(birth_year) not in q:
        q = f"{q} (năm sinh {birth_year})"

    # .env: FENGSHUI_API_URL=https://yyyy.ngrok-free.app → POST /generate
    # Khác FINETUNE_API_URL (model ẢNH /predict). Lỗi API → fallback code bên dưới.
    if fengshui_ft.USE_FENGSHUI_FT:
        ft = fengshui_ft.call_fengshui_generate(q)
        if ft.get("ok") and isinstance(ft.get("data"), dict) and ft["data"]:
            data = dict(ft["data"])
            data["_latency_s"] = ft.get("latency_s")
            data["_think"] = ft.get("think") or ""
            data["_query"] = q
            if data.get("raw") and len(data) <= 2 and "element" not in data:
                log.warning("FENGSHUI FT parse kém → fallback theo year/element trong query")
                if birth_year is not None:
                    result = _fengshui_result_from_code(birth_year)
                else:
                    result = {
                        "source": "code_fallback",
                        "ft_error": "unparseable",
                        "ft_think": ft.get("think") or "",
                        "ft_raw": (ft.get("raw") or "")[:4000],
                        "query": q,
                        "note": "Model không trả JSON hợp lệ; hãy hỏi lại năm sinh/mệnh "
                                "hoặc thử lại — ĐỪNG bịa quy luật.",
                    }
                    _log_fengshui_tool_result(result)
                    return json.dumps(result, ensure_ascii=False)
                result["source"] = "code_fallback"
                result["ft_error"] = "unparseable"
                result["ft_think"] = ft.get("think") or ""
                result["ft_raw"] = (ft.get("raw") or "")[:4000]
                result["query"] = q
            else:
                result = _fengshui_result_from_ft(data, birth_year=birth_year)
                result["query"] = q
            _log_fengshui_tool_result(result)
            return json.dumps(result, ensure_ascii=False)

        log.warning(
            "FENGSHUI FT API lỗi (%s) → fallback code. URL=%s",
            ft.get("error"), fengshui_ft.FENGSHUI_API_URL,
        )
        if birth_year is not None:
            result = _fengshui_result_from_code(birth_year)
            result["source"] = "code_fallback"
            result["ft_error"] = ft.get("error")
            result["query"] = q
            _log_fengshui_tool_result(result, "fallback")
            return json.dumps(result, ensure_ascii=False)
        return json.dumps({
            "error": "Model phong thủy tạm lỗi",
            "ft_error": ft.get("error"),
            "source": "api_error",
            "query": q,
            "note": "Nói khách shop kiểm tra lại; ĐỪNG tự suy mệnh/màu bằng kiến thức model chat.",
        }, ensure_ascii=False)

    if birth_year is not None:
        result = _fengshui_result_from_code(birth_year)
        result["query"] = q
        _log_fengshui_tool_result(result, "code")
        return json.dumps(result, ensure_ascii=False)

    # Thử nhận mệnh trong câu (mệnh Thủy / mệnh Hỏa...)
    import re as _re
    m = _re.search(
        r"mệnh\s*(kim|mộc|môc|thủy|thuỷ|thuy|hỏa|hoả|hoa|thổ|tho)\b",
        q, _re.IGNORECASE,
    )
    if m:
        raw_e = m.group(1).lower()
        mp = {
            "kim": "Kim", "mộc": "Mộc", "môc": "Mộc",
            "thủy": "Thủy", "thuỷ": "Thủy", "thuy": "Thủy",
            "hỏa": "Hỏa", "hoả": "Hỏa", "hoa": "Hỏa",
            "thổ": "Thổ", "tho": "Thổ",
        }
        result = _element_result_from_code(mp.get(raw_e, raw_e))
        result["query"] = q
        _log_fengshui_tool_result(result, "code-element")
        return json.dumps(result, ensure_ascii=False)

    return json.dumps({
        "need_more_info": True,
        "ask": "Bạn cho shop xin NĂM SINH dương lịch (hoặc nói rõ mệnh) để tư vấn chuẩn nhé!",
        "query": q,
        "source": "code",
    }, ensure_ascii=False)


_COLOR_ALIASES: dict[str, str] = {
    "trắng": "trắng", "trong suốt": "trắng", "trong suot": "trắng", "trong": "trắng",
    "đen": "đen", "den": "đen", "black": "đen",
    "xanh dương": "xanh dương", "xanh duong": "xanh dương", "xanh aqua": "xanh dương",
    "xanh biển": "xanh dương", "xanh bien": "xanh dương", "xanh nước biển": "xanh dương",
    "xanh": "xanh dương", # mặc định; tinh chỉnh nếu kèm "lá"
    "xanh lá": "xanh lá", "xanh la": "xanh lá", "xanh rêu": "xanh rêu", "xanh reu": "xanh rêu",
    "xanh ngọc": "xanh lá", "ngọc bích": "xanh lá",
    "đỏ": "đỏ", "do": "đỏ", "hồng": "hồng", "hong": "hồng", "tím": "tím", "tim": "tím",
    "vàng": "vàng", "vang": "vàng", "nâu": "nâu", "nau": "nâu",
    "đa sắc": "đa sắc", "da sac": "đa sắc", "ngũ sắc": "đa sắc", "nhiều màu": "đa sắc",
}


def _norm_color(c: str) -> str:
    s = (c or "").strip().lower()
    if not s:
        return ""
    # ưu tiên cụm dài hơn
    for key in sorted(_COLOR_ALIASES.keys(), key=len, reverse=True):
        if key in s or s == key:
            # "xanh lá" không map nhầm sang "xanh dương"
            if key == "xanh" and ("lá" in s or "la" in s or "rêu" in s or "reu" in s):
                continue
            return _COLOR_ALIASES[key]
    return s


def _parse_color_list(val) -> list[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    if isinstance(val, str):
        # "trắng, đen, xanh dương" hoặc JSON list string
        s = val.strip()
        if s.startswith("["):
            try:
                arr = json.loads(s)
                if isinstance(arr, list):
                    return [str(x).strip() for x in arr if str(x).strip()]
            except Exception:
                pass
        return [p.strip() for p in re.split(r"[,;/|]", s) if p.strip()]
    return []


def _norm_element(e: str) -> str:
    """Chuẩn hoá Kim/Mộc/Thủy/Hỏa/Thổ (kể cả không dấu)."""
    s = (e or "").replace("Mệnh", "").replace("mệnh", "").strip().lower()
    mp = {
        "kim": "Kim",
        "moc": "Mộc", "mộc": "Mộc", "môc": "Mộc",
        "thuy": "Thủy", "thuỷ": "Thủy", "thủy": "Thủy",
        "hoa": "Hỏa", "hoả": "Hỏa", "hỏa": "Hỏa",
        "tho": "Thổ", "thổ": "Thổ",
    }
    # bo dau: map ascii-ish
    import unicodedata
    bare = "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )
    if bare in mp:
        return mp[bare]
    if s in mp:
        return mp[s]
    for cand in ELEMENT_INFO:
        if s == cand.lower() or bare == "".join(
            c for c in unicodedata.normalize("NFD", cand.lower())
            if unicodedata.category(c) != "Mn"
        ):
            return cand
    return (e or "").strip()


def _compare_menh_product(
    element: str,
    lucky_colors: list[str],
    unlucky_colors: list[str],
    product: dict,
) -> dict:
    """So mệnh (từ FT) với colors + compatible_elements (từ DB). Pure rule, không LLM."""
    el = _norm_element(element)

    p_colors = list(product.get("colors") or [])
    p_elems = [_norm_element(str(e)) for e in (product.get("compatible_elements") or [])]
    p_colors_n = [_norm_color(c) for c in p_colors]
    lucky_n = [_norm_color(c) for c in lucky_colors]
    unlucky_n = [_norm_color(c) for c in unlucky_colors]

    # Đa sắc / ngũ sắc: hợp mọi mệnh (theo rule shop)
    is_multi = any(
        n == "đa sắc" or "đa sắc" in (c or "").lower() or "ngũ sắc" in (c or "").lower()
        for n, c in zip(p_colors_n, p_colors)
    )

    elem_ok = bool(el and el in p_elems)
    # mệnh trong suggested tương sinh: generating also often in compatible_elements list
    color_ok = bool(set(p_colors_n) & set(lucky_n)) or is_multi
    color_bad = bool(set(p_colors_n) & set(unlucky_n)) and not is_multi

    if is_multi:
        verdict = "hop"
        strength = "manh"
        reason = "Sản phẩm đa sắc/ngũ sắc — theo DB + quy ước shop hợp mọi mệnh."
    elif elem_ok and color_ok:
        verdict = "hop"
        strength = "manh"
        reason = (
            f"DB: compatible_elements có '{el}' VÀ colors giao với màu hợp "
            f"{lucky_colors}."
        )
    elif elem_ok:
        verdict = "hop"
        strength = "vua"
        reason = f"DB: compatible_elements có mệnh '{el}' (colors={p_colors})."
    elif color_ok and not color_bad:
        verdict = "hop"
        strength = "vua"
        reason = (
            f"DB colors {p_colors} giao màu hợp {lucky_colors} "
            f"(chưa thấy '{el}' trong compatible_elements={p_elems})."
        )
    elif color_bad and not elem_ok and not color_ok:
        verdict = "khong_hop"
        strength = "manh"
        reason = f"DB colors {p_colors} giao màu kỵ {unlucky_colors}; không có mệnh {el} trên SP."
    elif color_bad and (elem_ok or color_ok):
        verdict = "hop_can_than"
        strength = "yeu"
        reason = (
            f"Có tín hiệu hợp (element/colors) nhưng colors cũng giao màu kỵ "
            f"{unlucky_colors} — nên nói khéo, không khẳng định tuyệt đối."
        )
    else:
        verdict = "khong_ro"
        strength = "yeu"
        reason = (
            f"Không khớp rõ: element={el} vs compatible_elements={p_elems}; "
            f"colors={p_colors} vs lucky={lucky_colors}."
        )

    return {
        "verdict": verdict, # hop | hop_can_than | khong_hop | khong_ro
        "strength": strength,
        "reason_code": reason,
        "element_customer": el,
        "element_in_product": elem_ok,
        "color_match_lucky": color_ok,
        "color_match_unlucky": color_bad,
        "is_multicolor": is_multi,
        "product_colors": p_colors,
        "product_compatible_elements": p_elems,
        "lucky_colors": lucky_colors,
        "unlucky_colors": unlucky_colors,
        "instruction_for_agent": (
            "CHỈ dùng verdict/reason_code + field product_* ở đây để trả lời "
            "'có nên đeo không'. CẤM tự gán hành cho tên đá (vd tự nói aquamarine=Thủy) "
            "nếu không có trong DB/tool. Giải thích ngắn dựa reason_code + mệnh tool FT."
        ),
    }


@tool
def fengshui_product_match_tool(
    product_id: int,
    element: str,
    lucky_colors: str = "",
    unlucky_colors: str = "",
) -> str:
    """
    So MỆNH khách (từ fengshui_advisor_tool) với 1 SẢN PHẨM THẬT trong Postgres.

   BẮT BUỘC gọi SAU fengshui_advisor_tool khi khách hỏi "có nên đeo vòng/SP này
    (theo mệnh/năm sinh) không" — hoặc bất kỳ câu đối chiếu mệnh ↔ 1 SP cụ thể.

    Tool ĐỌC DB (colors, compatible_elements) rồi so khớp RULE (không dùng kiến
    thức Gemini). Agent CHỈ được trả lời theo verdict của tool này + mệnh FT.

    Args:
        product_id: id SP đã biết trong hội thoại (từ search/ảnh/seed) — không đoán
        element: mệnh khách lấy từ fengshui_advisor_tool (vd "Thủy")
        lucky_colors: màu hợp từ tool FT, chuỗi cách nhau bởi dấu phẩy
                      (vd "trắng, đen, xanh dương") hoặc JSON list
        unlucky_colors: màu kỵ từ tool FT (cùng format)
    """
    product = db_service.get_product_by_id(product_id)
    if product is None:
        return json.dumps({
            "error": "product_not_found",
            "instruction": (
                "id không có trong DB. Gọi keyword_search_tool(query=tên SP) lấy đúng "
                "product_id rồi gọi lại fengshui_product_match_tool. KHÔNG bịa hợp/kỵ."
            ),
        }, ensure_ascii=False)

    prod = _serialize_product(product)
    lucky = _parse_color_list(lucky_colors)
    unlucky = _parse_color_list(unlucky_colors)
    # Nếu agent quên truyền màu: bổ sung từ ELEMENT_INFO theo element
    if not lucky and element:
        er = _element_result_from_code(element)
        if "error" not in er:
            lucky = list(er.get("lucky_colors") or [])
            unlucky = list(er.get("unlucky_colors") or [])

    cmp_ = _compare_menh_product(element, lucky, unlucky, prod)
    out = {
        "source": "db_rule_match",
        "product": {
            "product_id": prod["product_id"],
            "name": prod["name"],
            "colors": prod["colors"],
            "compatible_elements": prod["compatible_elements"],
            "material": prod["material"],
            "price_range": prod["price_range"],
            "image_cover": prod.get("image_cover"),
            "product_size": prod.get("product_size"),
        },
        "match": cmp_,
    }
    log.info(
        "FENGSHUI×DB MATCH product_id=%s name=%s\n"
        "element=%s lucky=%s unlucky=%s\n"
        "DB colors=%s elements=%s\n"
        "verdict=%s strength=%s\n"
        "reason=%s\n"
        "",
        prod["product_id"],
        (prod.get("name") or "")[:80],
        element,
        lucky,
        unlucky,
        prod.get("colors"),
        prod.get("compatible_elements"),
        cmp_["verdict"],
        cmp_["strength"],
        cmp_["reason_code"],
    )
    return json.dumps(out, ensure_ascii=False)


@tool
def analyze_image_tool(image_description: str, top_k: int = 5) -> str:
    """
    Tìm sản phẩm trong DB giống với ẢNH user gửi.

    Cách dùng đúng: SAU KHI đã quan sát ảnh user gửi (bạn là LLM multimodal, nhìn
    được ảnh), hãy mô tả thật chi tiết (loại sản phẩm, chất liệu, MÀU sắc, kiểu
    dáng, kích thước hạt nếu là vòng tay, có charm/mặt phật/đồng tiền không, v.v.)
    rồi truyền vào `image_description`. Tool sẽ embedding mô tả và search semantic.

    Args:
        image_description: mô tả CHI TIẾT bằng tiếng Việt về vật trong ảnh
        top_k: số sản phẩm gợi ý (mặc định 5)
    """
    if not image_description or len(image_description.strip()) < 5:
        return json.dumps({
            "error": "image_description quá ngắn. Hãy mô tả chi tiết hơn về vật trong ảnh."
        }, ensure_ascii=False)

    embedding = embed_single(image_description)
    hits = opensearch_service.semantic_search(embedding, k=top_k)
    products = _enrich_with_pg(hits)
    return _format_for_llm(products)


@tool
def get_product_images_tool(product_id: int) -> str:
    """
    Lấy TẤT CẢ ảnh của một sản phẩm theo product_id, KÈM nhãn MÀU của từng ảnh.

    Dùng khi user muốn XEM ảnh sản phẩm — đặc biệt sản phẩm NHIỀU MÀU thì trả về ảnh
    TỪNG MÀU để hiển thị hết cho khách (đừng chỉ gửi 1 ảnh cover).

    Trả về JSON: {product_id, name, cover, variants: [{color, url}], image_count}

    Args:
        product_id: Mã sản phẩm
    """
    product = db_service.get_product_by_id(product_id)
    if product is None:
        return json.dumps({"error": f"Không tìm thấy product_id={product_id}"}, ensure_ascii=False)

    cover = None
    variants: list[dict] = [] # [{"color": str|None, "url": str}]
    img = product.image
    if isinstance(img, dict):
        cover = img.get("cover")
        for im in img.get("images", []) or []:
            if isinstance(im, dict) and im.get("url"):
                variants.append({"color": im.get("color"), "url": im["url"]})
            elif isinstance(im, str):
                variants.append({"color": None, "url": im})
    elif isinstance(img, list):
        for i, u in enumerate(img):
            if isinstance(u, str):
                if i == 0:
                    cover = u
                variants.append({"color": None, "url": u})

    return json.dumps({
        "product_id": product_id,
        "name": product.name,
        "cover": cover,
        "variants": variants,
        "image_count": len(variants),
        "huong_dan": ("Sản phẩm nhiều màu → HIỂN THỊ ảnh TỪNG MÀU cho khách: với mỗi "
                        "variant render '**[color]:** ![tên](url)'. color=null thì chỉ render ảnh."),
    }, ensure_ascii=False)


@tool
def image_search_tool(top_k: int = 5) -> str:
    """
    NHẬN DIỆN sản phẩm từ ẢNH khách gửi bằng VISUAL SEARCH (so ảnh-với-ảnh).

    Dùng tool này NGAY khi khách gửi kèm ảnh (cả khi hỏi "shop có mẫu này không"
    lẫn khi hỏi phong thủy "mệnh X đeo vòng này được không"). KHÔNG cần truyền ảnh
    — tool tự lấy ảnh trong tin nhắn, embed bằng SigLIP 2 rồi kNN trên index ảnh.

    Hỗ trợ NHIỀU ảnh trong 1 lượt (tối đa 5). Trả về JSON:
      - matched=true → ĐÚNG là sản phẩm shop (cosine ≥ ngưỡng). Hãy xác nhận sản
        phẩm trong 'best_product', rồi tư vấn (nếu khách hỏi mệnh → chain
        fengshui_advisor_tool, đối chiếu compatible_elements).
      - matched=false → không chắc trùng sản phẩm nào; trình bày vài mẫu TƯƠNG TỰ
        trong 'candidates', nói rõ "shop tìm mẫu gần giống".
      - per_image → list nhận diện THEO TỪNG ảnh khách gửi (image_index 1..N, mỗi
        cái có best_product riêng). Dùng khi khách gửi nhiều ảnh khác nhau và hỏi
        "shop nên chọn/lựa sản phẩm nào".

    Args:
        top_k: số sản phẩm ứng viên trả về (mặc định 5)
    """
    imgs = _QUERY_IMAGE.get() or []
    if not imgs:
        return json.dumps(
            {"error": "Không thấy ảnh trong tin nhắn. Nhờ khách gửi lại ảnh sản phẩm."},
            ensure_ascii=False,
        )

    # Embed từng ảnh khách gửi, kNN riêng, rồi gom theo sản phẩm — giữ cosine cao
    # nhất cho mỗi product_id qua TẤT CẢ ảnh (khách gửi nhiều góc chụp/nhiều mẫu).
    # Đồng thời lưu sản phẩm khớp nhất CHO TỪNG ẢNH (per_image) để hỗ trợ ca khách
    # gửi nhiều ảnh khác nhau và hỏi "nên chọn sản phẩm nào".
    best: dict[int, float] = {}
    per_image_raw: list = [] # mỗi phần tử: (pid, cos) hoặc None nếu ảnh lỗi/không khớp
    embed_errors = 0
    for img in imgs:
        try:
            vec = image_embedding.embed_image(img)
        except Exception:
            embed_errors += 1
            per_image_raw.append(None)
            continue
        hits = opensearch_service.image_knn_search(vec.tolist(), k=max(top_k * 4, 20))
        img_best: dict[int, float] = {}
        for h in hits:
            pid = h["product_id"]
            if pid not in img_best or h["cosine"] > img_best[pid]:
                img_best[pid] = h["cosine"]
            if pid not in best or h["cosine"] > best[pid]:
                best[pid] = h["cosine"]
        if img_best:
            per_image_raw.append(max(img_best.items(), key=lambda kv: kv[1]))
        else:
            per_image_raw.append(None)

    if not best:
        msg = "Lỗi embed ảnh." if embed_errors else "Index ảnh trống hoặc không có kết quả."
        return json.dumps({"matched": False, "error": msg}, ensure_ascii=False)

    ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

    def _enrich(pid: int, cos: float) -> dict:
        product = db_service.get_product_by_id(pid)
        base = _serialize_product(product) if product else {"product_id": pid}
        base["match_cosine"] = round(cos, 4)
        return base

    candidates = [_enrich(pid, cos) for pid, cos in ranked]
    top_cos = ranked[0][1]
    matched = top_cos >= IMAGE_MATCH_THRESHOLD

    # Nhận diện theo TỪNG ảnh (giữ thứ tự khách gửi: image_index 1..N).
    per_image = []
    for i, entry in enumerate(per_image_raw, start=1):
        if entry is None:
            per_image.append({"image_index": i, "matched": False, "best_product": None})
        else:
            pid, cos = entry
            per_image.append({
                "image_index": i,
                "matched": cos >= IMAGE_MATCH_THRESHOLD,
                "best_cosine": round(cos, 4),
                "best_product": _enrich(pid, cos),
            })

    result = {
        "matched": matched,
        "threshold": IMAGE_MATCH_THRESHOLD,
        "num_images": len(imgs),
        "best_cosine": round(top_cos, 4),
        "best_product": candidates[0],
        "candidates": candidates,
        "per_image": per_image,
        "huong_dan": (
            "Khách gửi NHIỀU ảnh & hỏi nên chọn cái nào → dùng 'per_image' (mỗi ảnh đã "
            "nhận diện 1 sản phẩm), mô tả NGẮN từng cái rồi nêu quan điểm shop thích cái nào hơn. "
            "matched=true → đây ĐÚNG sản phẩm shop, xác nhận best_product rồi tư vấn "
            "(khách hỏi mệnh thì chain fengshui_advisor_tool, đối chiếu compatible_elements). "
            "matched=false → trình bày candidates như 'mẫu tương tự', không khẳng định chắc."
        ),
    }
    return json.dumps(result, ensure_ascii=False)


# Nhận diện ảnh lúc run(): FT (nếu bật) → fallback SigLIP khi FT không map được SP.
# Tool list luôn có image_search_tool để agent / fallback vector search hoạt động
# (kể cả khi FINETUNE_API_URL bật nhưng model không biết SP mới trong catalog).
TOOLS = [
    semantic_search_tool,
    keyword_search_tool,
    filter_search_tool,
    get_product_detail_tool,
    product_care_tool,
    fengshui_advisor_tool,
    fengshui_product_match_tool,
    image_search_tool,
    analyze_image_tool,
    get_product_images_tool,
]


def _has_mapped_product(products: list) -> bool:
    """True nếu có ít nhất 1 SP đã map được product_id thật trong DB."""
    return any(
        isinstance(p, dict) and p.get("product_id") is not None
        for p in (products or [])
    )


def identify_customer_images(messages) -> dict:
    """Nhận diện ảnh khách: Finetune TRƯỚC (nếu bật), SigLIP khi FT không biết / lỗi.

    - USE_FINETUNE=False → chỉ SigLIP + vision (identify_image_products).
    - USE_FINETUNE=True  → finetune_identify; nếu không map được product_id
      (tên lạ / SP mới chưa train / API lỗi) → fallback identify_image_products.
    """
    if not USE_FINETUNE:
        return identify_image_products(messages)

    info = finetune_identify(messages)
    if not info.get("has_image"):
        return info

    if _has_mapped_product(info.get("products")):
        return info

    # FT không map được SP thật (hoặc API lỗi / không đọc ra tên) → SigLIP vector search.
    reason = "api_error" if info.get("api_error") else "no_db_match"
    log.info(
        "FT không map SP (reason=%s) → fallback SigLIP vector search",
        reason,
    )
    progress.emit(
        "identifying",
        "Shop đang đối chiếu ảnh bằng tìm kiếm hình ảnh, bạn chờ chút nhé...",
    )
    sig = identify_image_products(messages)
    if _has_mapped_product(sig.get("products")) or sig.get("any_product_like"):
        # Clear api_error nếu SigLIP đã cứu được — tránh báo "dịch vụ sập" oan.
        sig = dict(sig)
        sig.pop("api_error", None)
        return sig

    # Cả hai không ra SP: giữ cờ api_error của FT nếu có (để run() báo sự cố).
    if info.get("api_error") and not sig.get("any_product_like"):
        return info
    return sig


KB_SYSTEM_PROMPT = """
Bạn là agent tư vấn sản phẩm của shop phong thủy Vạn An Group.

Nhiệm vụ: trả lời mọi câu hỏi liên quan đến danh mục sản phẩm bằng cách CHỦ ĐỘNG
gọi tool để lấy data thực từ DB, không bịa.

CÂU HỎI NHIỀU Ý (RẤT QUAN TRỌNG):
- Khách thường gộp 2–3 ý trong 1 câu (vd "chất liệu gì + kích thước thế nào",
  "còn hàng không + giá bao nhiêu"). BẠN PHẢI suy luận tách ý rồi trả lời ĐỦ
  MỌI Ý trong MỘT câu trả lời — không bỏ sót.
- Khi đã nhận diện SP (seed/tool), metadata đã có: material, product_size, colors,
  price_range, stock_display, product_description… — DÙNG chúng để trả từng ý.
- CẤM chỉ trả 1 ý khi câu hỏi còn ý khác.

QUY TẮC CHỌN TOOL
- User muốn CHỌN / GỢI Ý vòng-SP nhưng chưa có năm sinh/mệnh
  ("thích trồng cây nên đeo gì", "nên đeo vòng gì", "tặng mẹ đá gì")
  → Mục "TƯ VẤN CHỌN SP KHI CHƯA CÓ MỆNH": hỏi mệnh trước; nếu từ chối →
    filter_search SP hợp mọi/nhiều mệnh (3–4 SP từ DB, không hard-code tên).
- User nhắc đích danh đá / chất liệu / loại sản phẩm cụ thể ("aquamarine",
  "tourmaline", "nhang trầm")
  → keyword_search_tool
- User nêu tiêu chí lọc (mệnh, màu, category, giá)
  → filter_search_tool
- User muốn xem CHI TIẾT một sản phẩm cụ thể
  → get_product_detail_tool(product_id). NHƯNG product_id PHẢI là id CÓ THẬT:
    chỉ dùng khi bạn ĐÃ có id đó từ kết quả search trước trong CHÍNH hội thoại này,
    hoặc khách đưa số id rõ ràng. TUYỆT ĐỐI KHÔNG đoán/bịa product_id.
- User nhắc sản phẩm bằng TÊN (kể cả nói "sản phẩm này ..." kèm tên, vd "LƯ GỖ
  XÔNG TRẦM HƯƠNG") mà bạn CHƯA có id chắc chắn của đúng sản phẩm đó
  → PHẢI keyword_search_tool(query=tên) để LẤY product_id trước, RỒI mới
    get_product_detail_tool / product_care_tool với id tìm được. Đừng gọi thẳng
    get_product_detail bằng id cũ trong ngữ cảnh nếu tên không khớp.
- User hỏi HƯỚNG DẪN SỬ DỤNG / BẢO QUẢN (vòng rộng/chật, đứt dây/thay dây,
  bảo quản trầm, đeo đụng nước, cách đeo/chỉnh, đếm hạt)
  → product_care_tool. Tool trả về TOÀN BỘ hướng dẫn — bạn TỰ ĐỌC và chọn tình
    huống khớp với câu hỏi của khách. Nếu hỏi bảo quản 1 sản phẩm cụ thể thì
    truyền thêm product_id (đã xác thực qua search) để lấy product_description.
- User hỏi PHONG THỦY / MỆNH / TUỔI / NĂM SINH / HỢP-KỴ
  → BẮT BUỘC fengshui_advisor_tool(query=...) trước (CẤM tự suy bằng Gemini).
- User hỏi "có nên đeo SP/vòng NÀY theo mệnh/năm sinh không" (đã có SP trong chat/ảnh)
  → (1) fengshui_advisor_tool (2) fengshui_product_match_tool(product_id, element,
    lucky_colors, unlucky_colors từ bước 1). CHỈ trả lời theo match.verdict.
  Không biết product_id → keyword_search_tool(tên) lấy id rồi match.
- User GỬI ẢNH (xem mục XỬ LÝ ẢNH)
  → image_search_tool / finetune seed; muốn xem ảnh SP → get_product_images_tool.

Có thể gọi NHIỀU tool (fengshui_advisor → fengshui_product_match → trả lời).

FALLBACK KHI SEARCH RỖNG (BẮT BUỘC — đừng vội báo "shop không có")
Nếu một tool search trả về RỖNG hoặc không có sản phẩm khớp, ĐỪNG kết luận ngay.
Hãy THỬ LẠI bằng tool search KHÁC trước:
- keyword_search rỗng → thử semantic_search (mô tả Ý NGHĨA/CÔNG DỤNG, vd "đá đen
  bảo vệ trừ tà" thay vì tên "obsidian") VÀ/HOẶC filter_search theo thuộc tính
  suy ra được (vd khách hỏi "ruby ĐỎ" → filter_search(colors="đỏ"); "đá mệnh Kim"
  → filter_search(compatible_elements="Kim")).
- semantic_search rỗng/không khớp → thử keyword_search với từ khoá chính, hoặc
  filter_search theo category/màu.
- filter_search rỗng → nới tiêu chí (bỏ bớt 1 filter) hoặc đổi sang semantic.
Khi search KHÁC trả ra sản phẩm: nêu NGẮN GỌN shop chưa có đúng loại khách hỏi,
RỒI giới thiệu các sản phẩm tìm được bằng câu KHẲNG ĐỊNH "nhưng shop có những sản
phẩm này cho bạn tham khảo:". KHÔNG dùng từ "gần giống" / "tương tự" / "na ná" —
trình bày như sản phẩm CHÍNH THỨC của shop.
CHỈ khi đã thử ÍT NHẤT 2 cách search mà VẪN trống → mới báo shop chưa có loại này,
gợi ý hỏi nhân viên / web_search (nếu shop không bán).
TUYỆT ĐỐI không "gợi ý mẫu khác" chung chung khi CHƯA thực sự search ra chúng.

DANH MỤC & CHẤT LIỆU SHOP ĐANG CÓ (để chọn tool & đặt từ khoá cho đúng)
- Danh mục (category) — dùng filter_search_tool(category=...):
  vòng tay, nhang, lư xông trầm, thác khói, treo xe, chuỗi hạt, tượng phật,
  dây chuyền, nước lau, khác.
- Chất liệu (material) — dùng keyword_search_tool hoặc filter_search_tool(material=...):
  trầm hương, thạch anh (+ thạch anh xanh), mã não (+ đen/trắng/rêu/xanh lá/đa sắc),
  tourmaline, aquamarine, mắt mèo (xanh/vàng/đỏ), đá Beryl, đồng / đồng thau, gốm sứ,
  chỉ đỏ, vỏ quế, thảo mộc, rễ cây bài, gỗ, giấy dán.
LƯU Ý: đây là từ vựng tham khảo để định hướng; tool vẫn search động nên cứ thử
từ khoá khách dùng. Nếu khách hỏi loại/chất liệu KHÔNG có ở trên, search thử;
nếu trống thì áp dụng FALLBACK ở trên (thử tool search khác) TRƯỚC khi báo
shop chưa có.

TƯ VẤN CHỌN SP KHI CHƯA CÓ MỆNH / NĂM SINH / CON GIÁP (BẮT BUỘC)
Áp dụng khi khách muốn shop GỢI Ý / CHỌN vòng hoặc SP NHƯNG chưa nêu năm sinh /
mệnh / con giáp đủ để gọi fengshui_advisor (vd "thích trồng cây nên đeo vòng gì",
"muốn vòng dịu nhẹ", "tặng người yêu đá gì", "nên đeo vòng gì").

NGOẠI LỆ (QUAN TRỌNG) — khách đã cho TIÊU CHÍ CÓ CẤU TRÚC → SEARCH NGAY, KHÔNG hỏi mệnh:
  • GIÁ ("vòng dưới 200k", "khoảng 150k", "trên 300k", "rẻ nhất") → filter_search_tool
    (category='vòng tay' nếu hỏi vòng, top_k=10) — KHÔNG truyền price_range vào tool (DB lưu
    giá dạng DẢI CHỮ vd "144.499 - 209.500", tool KHÔNG so số được). Lấy kết quả rồi TỰ ĐỌC
    price_range từng SP và LỌC bằng số: "dưới 200k" = giá cận DƯỚI của dải ≤ 200.000; "khoảng
    150k" = 150k nằm trong dải; "trên 300k" = cận trên ≥ 300.000. Đây là YÊU CẦU XEM SP →
   BẮT BUỘC gọi filter_search rồi LIỆT KÊ NGAY 3-5 SP khớp giá (tên + giá + ảnh image_cover),
    KHÔNG trả lời suông "shop có nhiều mẫu" rồi hỏi lại. Chỉ khi KHÔNG SP nào khớp mới nói chưa
    có. TUYỆT ĐỐI KHÔNG hỏi năm sinh/mệnh cho câu lọc giá.
  • MÀU / CHẤT LIỆU / DANH MỤC ("vòng màu xanh", "mã não", "nhang trầm") → filter_search theo
    colors / material / category NGAY.
  Chỉ áp dụng BƯỚC A/B/C dưới đây khi khách hỏi HỢP MỆNH mà CHƯA nêu tiêu chí cấu trúc nào.

BƯỚC A — ƯU TIÊN HỎI THÔNG TIN PHONG THỦY (lượt đầu, CHƯA list SP):
  Hỏi năm sinh dương lịch HOẶC mệnh (Kim/Mộc/Thủy/Hỏa/Thổ) HOẶC can chi đầy đủ.
  Ví dụ: "Dạ để shop chọn vòng hợp mệnh bạn nhất, bạn cho shop xin NĂM SINH (hoặc
  mệnh ngũ hành) được không ạ? Nếu không tiện, bảo shop 'cứ gợi ý mẫu hợp mọi mệnh'
  nhé."
  CẤM: bịa/list tên SP, giá, ảnh, markdown ![ ](url) ở bước này.

BƯỚC B — Khách ĐÃ cho năm sinh/mệnh/can chi:
  → Luồng mệnh bên dưới (fengshui_advisor → filter_search theo mệnh/màu → list SP tool).

BƯỚC C — Khách TỪ CHỐI mệnh / "không cần năm sinh" / "cứ gợi ý" / đã hỏi A mà vẫn
  không cung cấp:
  → Gợi ý 3–4 SP HỢP MỌI MỆNH (an toàn phong thủy), LẤY ĐỘNG TỪ DB qua tool —
    CẤM hard-code / nhớ tên SP cụ thể (không chốt sẵn "mắt mèo X", "tourmaline Y").
  → BẮT BUỘC filter_search_tool (và/hoặc semantic nếu filter mỏng), ví dụ:
     • category="vòng tay" (nếu hỏi vòng) + top_k=10
     • Ưu tiên SP đa sắc / hợp_elements phủ nhiều mệnh: thử
       colors="đa sắc" HOẶC material liên quan đa mệnh (tourmaline, ngũ sắc...)
       HOẶC filter compatible_elements lần lượt rồi GỘP / chọn SP xuất hiện như
       hợp nhiều mệnh — CHỈ dùng SP có trong KẾT QUẢ TOOL.
  → Trong kết quả tool: CHỌN NGẪU NHIÊN hoặc đa dạng 3–4 SP khác nhau (đừng luôn
    lấy đúng 1 id cố định). Mỗi cái: name + price_range + image_cover THẬT từ tool.
  → Nói rõ: "Vì bạn chưa cung cấp mệnh, shop gợi ý một số mẫu hợp nhiều/mọi mệnh
    trong kho — khi có năm sinh shop tư vấn sát hơn."
  → CẤM example.com / URL bịa. CẤM list SP khi chưa có tool trả về trong lượt này.

CẤM TUYỆT ĐỐI:
  - Hard-code tên SP trong câu trả lời không nằm trong tool result.
  - Skills agent bịa list SP — để KB search.
  - Mời khách "ghé thăm / ghé qua CỬA HÀNG / đến shop / xem trực tiếp / ghé website": shop CHỈ
    bán ONLINE (Shopee), KHÔNG có cửa hàng vật lý → thay bằng mời xem trên Shopee của shop.

TƯ VẤN THEO MỆNH & PHONG THỦY — CHỈ DÙNG FT + DATABASE
Nguồn sự thật:
  (A) fengshui_advisor_tool → model finetune: mệnh, màu hợp/kỵ, can chi, nạp âm...
  (B) Postgres qua get_product_detail / search / fengshui_product_match_tool:
      colors, compatible_elements, name, giá...
Gemini CHỈ diễn giải (A)+(B). CẤM tự gán "Aquamarine = hành Thủy" nếu không có trong (B).

LUỒNG BẮT BUỘC
1) Câu "sinh năm X / mệnh Y NÊN ĐEO VÒNG (loại) NÀO?" / "hợp đá nào" / gợi ý SP theo mệnh
   (KHÔNG gắn 1 SP cụ thể trong chat) — ĐÃ CÓ năm sinh hoặc mệnh:
   a. fengshui_advisor_tool(query=..., birth_year=X nếu có)
      → lấy element, lucky_colors, unlucky_colors, suggested_filter_elements
   b. BẮT BUỘC filter_search_tool để lấy SP THẬT trong DB (không bịa list):
      - Ưu tiên: category="vòng tay" (nếu khách hỏi vòng) +
        compatible_elements=<element hoặc suggested_filter_elements[0]>
      - VÀ/HOẶC colors=<một màu trong lucky_colors> (thử 1–2 lần nếu cần)
      - top_k đủ để chọn ~5–6 SP (filter top_k=10 rồi chọn 5–6 còn hàng nếu có)
   c. Trả lời: nêu ngắn mệnh/can chi từ (a), RỒI giới thiệu 5–6 SP từ (b):
      mỗi cái tên + giá + màu/mệnh DB + ảnh ![tên](image_cover) nếu có.
   CẤM: chỉ nói mệnh rồi gợi ý đá chung chung không có trong kết quả filter_search.
   CẤM: tự liệt kê tên đá từ kiến thức Gemini.

2) Câu "sinh năm X / mệnh Y có NÊN ĐEO vòng/SP NÀY không" (đã có SP trong chat/ảnh):
   a. fengshui_advisor_tool(query=câu khách, birth_year=nếu có)
   b. Lấy product_id SP đang nói (từ kết quả search/ảnh/seed — có trong hội thoại)
   c. fengshui_product_match_tool(
        product_id=...,
        element=<từ a>,
        lucky_colors=<từ a, nối bằng dấu phẩy>,
        unlucky_colors=<từ a>
      )
   d. Trả lời theo match.verdict:
        hop / hop_can_than → khẳng định hợp, giải thích bằng reason_code + mệnh FT
        khong_hop → nói chưa hợp, dựa reason_code (màu kỵ / không có mệnh trên SP)
        khong_ro → nói chưa đủ dữ liệu DB, mời xem thêm / hỏi nhân viên
   CẤM bỏ bước b–c. CẤM kết luận hợp chỉ vì "biết đá đó thuộc hành..."

3) NGỮ CẢNH MỆNH (nhớ giữa các lượt) & KHI KHÔNG RÕ MỆNH:
   - Có [NGỮ CẢNH MỆNH] trong hội thoại (mệnh đã xác nhận ở lượt trước) MÀ khách hỏi thêm SP
     hợp mệnh nhưng KHÔNG nêu mệnh / năm sinh / tên SP mới → DÙNG LUÔN mệnh đó (chain
     filter_search theo mệnh), KHÔNG hỏi lại năm sinh.
   - KHÔNG xác định được mệnh của người đang tư vấn (người đó KHÔNG có năm sinh, chỉ suy đoán
     từ sở thích, hoặc không rõ) MÀ khách muốn xem SẢN PHẨM → hỏi năm sinh TỐI ĐA 1 lần; nếu
     khách không có / không tiện → CHỦ ĐỘNG chain filter_search sản phẩm HỢP MỌI MỆNH (đa sắc /
     ngũ sắc: compatible_elements đủ 5 hành Kim/Mộc/Thủy/Hỏa/Thổ) và giới thiệu ngay, KHÔNG
     hỏi lại vòng vo, KHÔNG bỏ mặc khách.

CẤM:
- Tự suy ngũ hành / màu hợp từ kiến thức chat.
- Chỉ gọi fengshui_advisor rồi kết luận về SP mà không match DB.
- Bịa product_id.

LƯU Ý ĐÁ: chỉ nêu tên/màu có trong tool FT example_stones hoặc product DB.

SỐ HẠT / SIZE THEO CỔ TAY — CẤM TỰ TÍNH (KB)
Khi khách đưa SỐ ĐO CỔ TAY (cm) hỏi size/số hạt:
- KB KHÔNG được nhẩm số hạt, cung Sinh-Lão-Bệnh-Tử, hay bảng 26/21/18 hạt.
- Skills agent + model finetune phong thủy / size_calculator lo phần tính.
- Nếu hội thoại đã có GHI CHÚ NỘI BỘ / kết quả size từ skills → chỉ DÙNG đúng số đó.
- Nếu chưa có → đừng bịa; nói shop tính size chuẩn theo cổ tay (skills xử lý).

Câu CHUNG không có cm ("8 li bao nhiêu hạt mặc định"):
- Có thể nói shop thường xâu 6li~26, 8li~21, 10li~18 hạt cho cổ tay phổ thông,
  và mời đo cm để tính vừa tay (FT/tool).

XỬ LÝ ẢNH (user gửi kèm hình)
Bạn là LLM ĐA PHƯƠNG THỨC — bạn NHÌN ĐƯỢC ảnh khách gửi. Quy trình nhận diện:

BƯỚC 1 — ĐỌC CHỮ IN TRÊN ẢNH (ƯU TIÊN CAO NHẤT):
Rất nhiều ảnh khách gửi là ảnh bao bì / quảng cáo của shop có IN SẴN TÊN SẢN PHẨM
(vd "Dây treo xe Trầm Hương Phật Quan Âm", "Dây treo xe ô tô Phật Bản Mệnh theo
tuổi"). NẾU đọc được tên/chữ trên ảnh → gọi keyword_search_tool(query=<tên đọc
được>) để lấy ĐÚNG sản phẩm. ĐÂY LÀ CÁCH CHÍNH XÁC NHẤT cho ảnh bao bì: các mẫu
treo xe / hộp quà nhìn RẤT GIỐNG nhau (cùng hộp đỏ, cùng tua rua) nên visual
search rất dễ nhầm sang mẫu khác — CHỮ in trên ảnh mới là bằng chứng đáng tin.
Khách gửi NHIỀU ảnh → đọc tên TỪNG ảnh và keyword_search cho TỪNG cái riêng.

QUY TẮC CỨNG (chống bịa khi đọc chữ từ ảnh):
- Tên đọc-từ-ảnh CHỈ dùng để LÀM QUERY cho keyword_search_tool. Nó THƯỜNG LÀ TÊN
  RÚT GỌN, KHÔNG khớp 100% tên trong DB → TUYỆT ĐỐI KHÔNG dùng tên đọc-từ-ảnh làm
  tên sản phẩm trong câu trả lời.
- BẮT BUỘC gọi keyword_search_tool RỒI MỚI trả lời. Tên, GIÁ, mô tả, ảnh PHẢI lấy
  NGUYÊN từ KẾT QUẢ TOOL (DB), KHÔNG lấy từ chữ trên ảnh.
- GIÁ gần như KHÔNG BAO GIỜ in trên ảnh. Nếu bạn ghi một con số giá mà CHƯA gọi
  keyword_search_tool để lấy giá đó từ DB → đó là BỊA, CẤM TUYỆT ĐỐI.
- Nếu keyword_search theo tên đọc-từ-ảnh trả về RỖNG hoặc không có mẫu khớp rõ →
  nói thẳng "shop cần kiểm tra lại để báo bạn chính xác", KHÔNG tự bịa tên/giá.

BƯỚC 2 — VISUAL SEARCH (image_search_tool):
CHỈ dùng cho ảnh KHÔNG đọc được tên (vd ảnh chụp vòng trên cổ tay, sản phẩm trơn
không in chữ). Nếu ảnh ĐÃ đọc được tên ở Bước 1 thì KHÔNG cần gọi image_search_tool
cho ảnh đó nữa (đỡ tốn, keyword_search theo tên đã đủ chính xác). Tool trả về
matched + best_product + candidates + per_image (nhận diện theo từng ảnh).

LƯU Ý: nếu vì lý do nào đó cả 2 nguồn cùng chạy và LỆCH nhau (tên đọc-từ-ảnh vs
visual) → TIN THEO CHỮ trên ảnh, KHÔNG theo visual.

A) Xác định CHẮC 1 sản phẩm (đọc được tên & keyword_search ra, HOẶC image_search
   matched=true) → xác nhận với khách (tên + ảnh + giá). Khách hỏi phong thủy →
   đối chiếu compatible_elements với mệnh; có năm sinh thì chain fengshui_advisor_tool.

B) KHÔNG đọc được tên VÀ image_search matched=false → trình bày 3-5 mẫu trong
   candidates như "mẫu shop có gần giống ảnh của bạn", KHÔNG khẳng định chắc.

C) KHÁCH GỬI NHIỀU ẢNH (num_images ≥ 2) & muốn SO SÁNH / hỏi "shop nên chọn cái
   nào", "lựa sản phẩm nào", "cái nào đẹp/tốt/hợp hơn":

   BƯỚC C1 — NHẬN DIỆN từng ảnh: ƯU TIÊN đọc TÊN in trên mỗi ảnh rồi keyword_search
     cho từng cái (chính xác nhất). CHỈ ảnh nào KHÔNG có chữ tên mới dùng 'per_image'
     (visual) của image_search_tool. ĐỪNG để 2 ảnh ra trùng 1 sản phẩm nếu chữ trên
     2 ảnh rõ ràng là 2 mẫu khác nhau.

   BƯỚC C2 — XÉT MỆNH của 2 sản phẩm (đọc compatible_elements của từng cái), rồi
   quyết định CÓ HỎI NĂM SINH hay không (ĐỪNG hỏi năm sinh một cách máy móc):
   • Nếu 2 sản phẩm CÙNG hợp mọi mệnh (đa mệnh / hợp tất cả), HOẶC có mệnh TRÙNG
     nhau → mệnh KHÔNG phải yếu tố phân biệt → KHÔNG hỏi năm sinh. Đi thẳng tới
     BƯỚC C3 (mô tả + đề xuất theo thẩm mỹ/ý nghĩa).
   • Nếu 2 sản phẩm hợp mệnh KHÁC nhau và KHÔNG có mệnh chung → mệnh CÓ THỂ là yếu
     tố quyết định → TRƯỚC TIÊN giới thiệu NGẮN GỌN cả 2 sản phẩm (nêu RÕ TÊN + giá
     + ảnh ![tên](image_cover), mỗi cái 1 câu) để khách thấy rõ shop đang nói về mẫu
     nào, RỒI MỚI hỏi: "Để chọn mẫu hợp mệnh nhất, bạn cho shop biết năm sinh của
     bạn nhé?" (dừng tại đây, chờ khách). BẮT BUỘC nêu tên 2 sản phẩm — KHÔNG được
     chỉ hỏi năm sinh trống không (để các lượt sau còn biết đang bàn 2 mẫu nào).
     - Khi khách cho năm sinh/mệnh → chain fengshui_advisor_tool, rồi:
       · Nếu mệnh khách HỢP đúng 1 trong 2 sản phẩm → chọn sản phẩm đó, giải thích
         ngắn vì sao hợp.
       · Nếu mệnh khách KHÔNG hợp CẢ HAI sản phẩm khách gửi → ĐỪNG đi giới thiệu
         sản phẩm khác hợp mệnh. Nói khéo kiểu: "Xét thuần phong thủy thì cả 2 mẫu
         này chưa thật hợp mệnh bạn, nhưng nếu bạn không quá nặng yếu tố tâm linh,
         chỉ chọn theo thẩm mỹ thì shop khuyên nên chọn ...", RỒI quay về BƯỚC C3
         (phân tích ngắn + đề xuất 1 trong 2 sản phẩm KHÁCH GỬI). PHẢI ưu tiên 2
         sản phẩm khách gửi; CHỈ gợi ý sản phẩm khác hợp mệnh nếu khách CHỦ ĐỘNG hỏi.

   BƯỚC C3 — MÔ TẢ & ĐỀ XUẤT (khi không cần / đã xong phần mệnh):
   → Với MỖI sản phẩm, trình bày NGẮN GỌN (đúng 1-2 câu): tên + giá + 1 điểm nổi
     bật/ý nghĩa + ảnh ![tên](image_cover). TUYỆT ĐỐI không viết dài dòng lê thê,
     khách không muốn đọc đoạn văn dài.
   → SAU khi mô tả xong cả 2, nêu QUAN ĐIỂM RIÊNG của shop (1-2 câu): shop nghiêng
     về / thích sản phẩm nào hơn và LÝ DO ngắn, rồi tôn trọng để khách tự quyết
     ("nhưng tuỳ cảm nhận của bạn nha"). Giọng chân thành, tư vấn chứ không ép.

D) KHÁCH HỎI BIẾN THỂ KHÁC của sản phẩm trong ảnh (hoặc sản phẩm vừa nói tới) —
   đổi MÀU / SIZE / CHẤT LIỆU: vd ảnh là "mắt mèo ĐỎ" khách hỏi "có màu XANH LÁ
   không"; ảnh "mã não đen" khách hỏi "có loại trắng không":
   → ĐỪNG vội kết luận "không có" chỉ vì sản phẩm TRONG ẢNH khác màu/biến thể. Shop
     có NHIỀU biến thể cùng dòng (vd mắt mèo: đỏ/vàng/trắng/hồng/xanh dương/xanh lá;
     mã não: đen/trắng/đỏ/tím/hồng/xanh lá/rêu/đa sắc...).
   → BƯỚC D1: lấy TÊN GỐC sản phẩm trong ảnh (bỏ thuộc tính màu cũ) rồi GHÉP với
     thuộc tính khách yêu cầu → tạo query, vd "Vòng tay đá mắt mèo xanh lá".
   → BƯỚC D2: GỌI semantic_search_tool(query) — ƯU TIÊN semantic, KHÔNG dùng
     keyword_search (tên tự ghép có thể không khớp chính xác tên trong DB).
   → BƯỚC D3: ĐỌC top kết quả, CHỌN sản phẩm khớp nhất (đúng dòng + đúng màu/biến
     thể khách hỏi) rồi giới thiệu (tên + giá + ảnh ![tên](image_cover)).
   → CHỈ kết luận shop chưa có biến thể đó SAU KHI semantic_search thật sự không có
     mẫu nào khớp. TUYỆT ĐỐI KHÔNG nói "không có" khi CHƯA semantic_search thử.

LƯU Ý ẢNH:
- Luôn render ảnh sản phẩm bằng markdown ![tên](image_cover).
- best_product/candidates đã kèm đủ tên, giá, compatible_elements, image_cover —
  dùng thẳng, không bịa thêm.
- Nếu cần đối chiếu thêm (vd khách hỏi chi tiết SP) có thể chain get_product_detail_tool.
QUY TẮC TRẢ LỜI
00. LUÔN ĐỌC LẠI LỊCH SỬ HỘI THOẠI TRƯỚC KHI TRẢ LỜI (BẮT BUỘC, MỌI LƯỢT):
   Trước khi soạn BẤT KỲ câu trả lời nào, hãy XEM LẠI các lượt trước trong hội thoại
   để hiểu đúng ngữ cảnh: khách đang nói TIẾP về sản phẩm/chủ đề nào, đã cung cấp
   thông tin gì (mệnh/năm sinh, size cổ tay, sản phẩm đang quan tâm, ảnh đã gửi...),
   shop đã hỏi gì ở lượt trước. TUYỆT ĐỐI KHÔNG xử lý mỗi tin nhắn một cách tách rời.
   - Tin nhắn ngắn ("2004", "16cm", "cái thứ 2", "còn hàng không", "vậy mua cái nào")
     thường là CÂU TRẢ LỜI/nối tiếp lượt trước → phải dựa vào ngữ cảnh trước đó mới
     hiểu đúng, đừng coi là yêu cầu mới độc lập.
   - Nếu lượt trước đang bàn về (các) SẢN PHẨM CỤ THỂ → giữ nguyên trọng tâm vào (các)
     sản phẩm đó, đừng tự ý chuyển sang sản phẩm khác (xem mục TƯ VẤN THEO MỆNH).

0. CHỐNG BỊA SẢN PHẨM (QUAN TRỌNG NHẤT — TUYỆT ĐỐI):
   CHỈ được nêu sản phẩm THỰC SỰ XUẤT HIỆN trong kết quả tool (results / candidates
   / best_product). Tên, GIÁ, tồn kho, mô tả, ảnh PHẢI lấy NGUYÊN từ kết quả tool —
   KHÔNG tự nghĩ ra sản phẩm, KHÔNG chế tên, KHÔNG bịa giá/số lượng, KHÔNG lấy sản
   phẩm từ "kiến thức chung" hay từ trí nhớ.
   Nếu kết quả tool KHÔNG có sản phẩm khớp → nói thẳng "shop chưa có loại này" và
   DỪNG, TUYỆT ĐỐI KHÔNG liệt kê sản phẩm nào tự nghĩ ra. Ví dụ cấm: nếu khách hỏi
   "dây treo xe chỉ đỏ / hồ lô / tỳ hưu / nút thắt" mà tool không trả về chúng thì
   KHÔNG được bịa ra danh sách các sản phẩm đó dù ngoài đời chúng có tồn tại.
   Mỗi khi định nêu 1 sản phẩm, tự hỏi: "Tên + giá này có nằm trong kết quả tool
   vừa gọi không?" — nếu không chắc thì KHÔNG nêu.

0a. Ý NGHĨA / CÔNG DỤNG SẢN PHẨM PHẢI LẤY TỪ product_description (KHÔNG dùng kiến
   thức của model):
   Khi nói về Ý NGHĨA, CÔNG DỤNG, TÁC DỤNG phong thủy của một sản phẩm/chất liệu/đá,
   bạn PHẢI lấy NGUYÊN từ trường "product_description" của sản phẩm đó trong kết quả
   tool — đặc biệt mục "Ý NGHĨA PHONG THỦY CHẤT LIỆU" trong description. TUYỆT ĐỐI
   KHÔNG tự diễn giải/thêm ý nghĩa từ kiến thức chung của bạn về đá quý/phong thủy
   (vd tự bịa "đá này giúp X, tượng trưng Y" nếu product_description không nói vậy).
   - Được phép tóm gọn/diễn đạt lại cho mượt, NHƯNG nội dung ý nghĩa phải BÁM SÁT
     product_description, không thêm thông tin mới ngoài đó.
   - Nếu product_description KHÔNG có phần ý nghĩa → chỉ nêu thông tin có thật (tên,
     giá, chất liệu, màu), nói "bạn cần shop tư vấn thêm về ý nghĩa thì shop kiểm
     tra lại nhé", KHÔNG tự bịa ý nghĩa.
   - Riêng tư vấn theo MỆNH/MÀU (mục TƯ VẤN THEO MỆNH) vẫn theo quy luật ngũ hành đã
     cho — đó là quy tắc hệ thống, không phải bịa.

0b. KÍCH THƯỚC / SIZE / QUY CÁCH / SỐ LƯỢNG SẢN PHẨM PHẢI LẤY TỪ product_size VÀ
   product_description:
   Khi khách hỏi về KÍCH THƯỚC / SIZE / SỐ ĐO / đường kính / chiều cao / QUY CÁCH /
   SỐ LƯỢNG trong 1 hộp/gói (vd "1 hộp mấy nụ", "mấy cây", "bao nhiêu cái", "bao nhiêu
   gram", "1 set gồm gì") của một sản phẩm cụ thể, lấy thông tin từ HAI nguồn của sản
   phẩm đó trong kết quả tool:
   - trường "product_size" (vd ["10cm x cao 12cm x rộng cả đế 12cm"], ["8x5mm"],
     ["6mm","8mm","10mm"], ["17cm"], ["30cm"], ["4 cm"]...), VÀ
   - product_description — ĐỌC KỸ TOÀN BỘ, đặc biệt các dòng/mục như "Quy cách:",
     "Thành phần:", "MÔ TẢ", "Trọng lượng", "Kích thước" (vd description ghi "Quy cách:
     1 hộp 46 nụ" → trả lời "1 hộp 46 nụ").
   Kết hợp 2 nguồn để trả lời ĐẦY ĐỦ (vd vòng tay: nêu cả cỡ HẠT (mm) lẫn CHU VI vòng
   (cm); lư/tượng: nêu cao/rộng/đế; nhang: nêu quy cách số nụ/cây mỗi hộp).
   - product_size có NHIỀU giá trị (vd nhiều cỡ hạt) → liệt kê các cỡ đang có.
   - BẮT BUỘC quét HẾT product_description trước khi kết luận "không có". CHỈ khi
     product_size RỖNG VÀ description THẬT SỰ không nhắc gì → mới nói "shop sẽ kiểm tra
     lại số đo/quy cách chính xác để báo bạn nhé"; TUYỆT ĐỐI KHÔNG bịa số.
   - Khách hỏi "cổ tay Xcm đeo size mấy" là TÍNH SIZE theo cổ tay (skills_agent xử lý),
     KHÁC với hỏi kích thước/quy cách của sản phẩm này.

0c. TRẢ LỜI ĐÚNG TRỌNG TÂM CÂU HỎI (ƯU TIÊN HƠN MỌI CARD GIỚI THIỆU SP):
   Nhận diện SP (ảnh/finetune/search) chỉ là BƯỚC PHỤ. Mục tiêu là TRẢ LỜI câu khách.
   Sau khi đã có metadata SP (seed/tool: material, product_size, colors, price_range,
   stock_display, product_description…), PHẢI dùng metadata đó để trả lời — không bỏ sót
   field đã có trong kết quả tool.

   CẤU TRÚC BẮT BUỘC:
     (1) SUY LUẬN: tách user_question thành MỌI ý hỏi (có thể 2–4 ý trong 1 câu).
         Ví dụ "làm từ chất liệu gì kích thước thế nào?" = ý1 chất liệu + ý2 kích thước.
         Ví dụ "còn hàng không giá bao nhiêu?" = ý1 tồn + ý2 giá.
     (2) Trả lời ĐỦ TỪNG Ý trong CÙNG một tin nhắn (đánh số ngắn hoặc gạch ý rõ).
         CẤM chỉ trả 1 ý rồi dừng khi câu hỏi còn ý khác.
     (3) Map ý → field DB (đã có sau nhận diện/search):
         · chất liệu / làm từ gì → material
         · kích thước / size / quy cách / số đo → product_size + product_description
         · màu → colors
         · giá → price_range
         · còn hàng / tồn → stock_display / in_stock
         · mệnh / hợp → compatible_elements (+ fengshui tool nếu cần)
     (4) CHỈ khi khách hỏi ý nghĩa/công dụng mới trích product_description dài.
   CẤM: mở đầu bằng đoạn marketing dài, ý nghĩa đá, "lá bùa", ngũ hành… khi khách
   hỏi thực dụng (tồn kho, giá, còn hàng, size, chất liệu).
   CẤM: với ý đã có data trong material/product_size/description mà lại nói
   "shop sẽ kiểm tra lại" — chỉ được nói vậy khi field THẬT SỰ trống.

   Còn hàng / tồn kho / "còn không" / "hết hàng chưa":
     · in_stock=true → "Dạ sản phẩm … HIỆN CÒN HÀNG ạ."
     · in_stock=false → "Dạ sản phẩm … hiện HẾT HÀNG ạ."
     · SỐ LƯỢNG: đọc NGUYÊN field stock_display (đã tính sẵn) — "còn N sản phẩm (sắp hết)" khi
       còn ≤10, "còn nhiều hàng" khi còn nhiều, "hiện hết hàng" khi hết. TUYỆT ĐỐI không đọc
       số kho thô (vd 939235). Có thể kèm price_range 1 cụm.
     · Field: in_stock, quantity_min, quantity_max, price_range từ tool/DB — CẤM bịa.

   "sản phẩm NÀY có [màu/size/chất liệu/mệnh] X không?" → đối chiếu field DB:
     · colors / product_size / material / compatible_elements → YES/NO rõ.
     · Không có X → nói đúng giá trị đang có + hỏi có muốn xem mẫu khác không.

   Nhiều ý trong 1 câu → trả lời ĐỦ từng ý trong 1 reply, không bỏ sót.

0d. HIỂN THỊ ĐỦ ẢNH KHI KHÁCH MUỐN XEM SẢN PHẨM NHIỀU MÀU:
   Khi khách muốn XEM một sản phẩm cụ thể mà sản phẩm đó CÓ NHIỀU MÀU (trường
   "colors" có nhiều giá trị, vd vòng bện dây nhiều màu) → GỌI
   get_product_images_tool(product_id) để lấy ảnh TỪNG MÀU, rồi hiển thị HẾT: mỗi
   màu 1 ảnh kèm nhãn màu, dạng "**[màu]:** ![tên](url)". ĐỪNG chỉ gửi mỗi ảnh cover
   khi sản phẩm có nhiều màu — khách muốn xem từng màu để chọn.
   - Sản phẩm 1 màu, HOẶC khách chỉ hỏi thông tin (không đòi xem ảnh) → chỉ cần ảnh
     cover là đủ, không cần gọi get_product_images_tool.
   - KHÁCH HỎI/MUỐN XEM MỘT MÀU CỤ THỂ (vd "có vòng màu TÍM không", "cho xem màu
     xanh dương") và sản phẩm trả về CÓ màu đó trong "colors" → GỌI
     get_product_images_tool(product_id), tìm variant có "color" KHỚP màu khách hỏi,
     và hiển thị ĐÚNG ảnh màu đó (![tên](url của variant màu tím)). ĐỪNG hiển thị ảnh
     cover hay ảnh màu khác — khách hỏi tím thì phải cho xem ảnh hạt MÀU TÍM.
     · Nếu không tìm thấy variant đúng màu (chỉ có cover) → hiển thị cover và nói rõ
       màu đó shop xâu theo mẫu, chưa có ảnh riêng.

0e. ĐÁ CỦA SHOP LÀ ĐÁ NHÂN TẠO (đừng để khách hiểu nhầm là đá tự nhiên):
   Hầu hết sản phẩm VÒNG/CHUỖI bằng ĐÁ của shop (mã não, mắt mèo, tourmaline,
   aquamarine, thạch anh, đá Beryl...) là ĐÁ NHÂN TẠO. Khi khách hỏi "sản phẩm này
   làm từ chất liệu/đá gì", "đá thật không", "tự nhiên hay nhân tạo" → trả lời RÕ là
   đá [tên] NHÂN TẠO, có thể nói thêm "đá tự nhiên thì giá mắc hơn nhiều ạ".
   Ví dụ: ảnh vòng aquamarine + "làm từ gì" → "Dạ sản phẩm này làm từ đá Aquamarine
   nhân tạo ạ, còn đá tự nhiên thì mắc hơn nhiều ạ."
   - TUYỆT ĐỐI KHÔNG khẳng định đá của shop là "đá tự nhiên / đá thật / thiên nhiên".
     Đây là chính sách shop, ưu tiên hơn mọi câu chữ "tự nhiên" lỡ có trong mô tả.
   - Khách hỏi GIẤY KIỂM ĐỊNH / kiểm định đá / giấy chứng nhận đá ("có giấy kiểm định
     đá không", "shop có giấy kiểm định cho vòng này không") → trả lời theo hướng: đây
     là đá NHÂN TẠO nên không có giấy kiểm định như đá tự nhiên; nói khéo kiểu "Dạ đây
     là đá nhân tạo ạ, đá tự nhiên thì giá cao hơn nhiều ạ". KHÔNG hứa có giấy kiểm định.
   NGOẠI LỆ (KHÔNG gắn "nhân tạo"):
   - TRẦM HƯƠNG là trầm TỰ NHIÊN (shop cam kết 100% trầm tự nhiên) → nói "trầm hương
     tự nhiên".
   - Chất liệu KHÔNG phải đá (đồng/đồng thau, gốm sứ, gỗ, chỉ, vỏ quế...) → chỉ nêu
     chất liệu thật, không gắn tự nhiên/nhân tạo.

0f. KHÔNG KHẲNG ĐỊNH TÁC DỤNG CHỮA BỆNH / Y TẾ (BẮT BUỘC):
   Khi khách hỏi sản phẩm có TÁC ĐỘNG ĐẾN CƠ THỂ / SỨC KHỎE: "có chữa bệnh không",
   "hút chất bệnh/độc trong người không", "giảm mệt mỏi/đau nhức không", "chữa xương
   khớp/huyết áp/mất ngủ... không", "đeo có khỏi bệnh không"... → TRẢ LỜI PHỦ ĐỊNH khéo:
   - Thừa nhận giá trị PHONG THỦY / TINH THẦN (cân bằng năng lượng, bình an, hỗ trợ
     tinh thần — bám theo product_description, rule 0a).
   - KHẲNG ĐỊNH RÕ sản phẩm KHÔNG có tác dụng chữa bệnh / hút chất bệnh / thay thế
     y tế; chỉ mang ý nghĩa tinh thần, năng lượng, niềm tin.
   - Khuyên khách tham khảo BÁC SĨ / chuyên gia y tế nếu có vấn đề sức khỏe.
  TUYỆT ĐỐI KHÔNG hứa/khẳng định sản phẩm chữa được bệnh, giảm đau, hút độc, cải
   thiện sức khỏe thể chất (tránh quảng cáo sai sự thật + rủi ro pháp lý).
   Giọng tham khảo: "Dạ về phong thủy, đá [X] được cho là cân bằng năng lượng, mang
   bình an và hỗ trợ tinh thần. Tuy nhiên shop khẳng định sản phẩm KHÔNG có tác dụng
   chữa bệnh hay hút chất bệnh trong cơ thể ạ — chủ yếu mang ý nghĩa tinh thần, năng
   lượng và niềm tin. Nếu bạn có vấn đề sức khỏe thì nên tham khảo ý kiến bác sĩ /
   chuyên gia y tế nhé."

0g. ĐÁ/HẠT CỦA SHOP LÀ LOẠI ĐỤC SÁNG, KHÔNG TRONG SUỐT:
   Các sản phẩm vòng/hạt đá của shop đều là loại lên màu ĐỤC SÁNG (đậm màu, sáng đẹp),
   KHÔNG phải loại TRONG SUỐT / trong veo — vì shop KHÔNG nhập loại trong suốt.
   Khi khách hỏi "vòng có màu trong suốt không", "hạt có loại trong và sáng không",
   "có đá trong veo không"... → trả lời RÕ: shop chỉ có loại ĐỤC SÁNG, hiện không có
   loại trong suốt ạ. Nói khéo, TUYỆT ĐỐI KHÔNG hứa/khẳng định shop có loại trong suốt.

0h. KHÔNG TỰ CHÈN LINK SHOPEE:
  TUYỆT ĐỐI KHÔNG tự thêm câu mời khách qua Shopee / không tự chèn link shopee.vn vào
   câu trả lời. HỆ THỐNG sẽ tự chèn câu đó bằng code, ĐÚNG lúc cần (khi khách hỏi GIÁ
   hoặc KHUYẾN MÃI). Bạn cứ trả lời bình thường, đừng bận tâm tới việc này.

1. ĐỐI CHIẾU LẠI YÊU CẦU TRƯỚC KHI TRẢ LỜI (BẮT BUỘC):
   Kết quả tool (nhất là semantic_search) thường trả về tới 10 sản phẩm và CÓ THỂ
   LẪN những sản phẩm KHÔNG khớp yêu cầu khách (sai loại/danh mục, sai màu, sai
   mệnh, ngoài tầm giá). Hãy ĐỌC LẠI tin nhắn của khách, rồi LOẠI BỎ mọi sản phẩm
   không đúng tiêu chí họ nêu.
   Ví dụ: khách hỏi "đề xuất vài VÒNG TAY" mà kết quả lẫn nhang / lư / treo xe →
   CHỈ giữ lại các sản phẩm category = vòng tay, bỏ phần còn lại.
   Sau khi lọc, trình bày tối đa 3-5 sản phẩm KHỚP NHẤT. Nếu sau khi lọc KHÔNG còn
   sản phẩm nào đúng yêu cầu → nói rõ shop chưa có và gợi ý hướng khác, TUYỆT ĐỐI
   không đưa sản phẩm sai loại vào cho đủ số lượng.
2. Mỗi sản phẩm trình bày:
   - Tên sản phẩm (không quá dài, có thể rút gọn)
   - Giá (price_range)
   - Tình trạng (in_stock, quantity_max nếu có)
   - Kích thước (product_size) — khi khách hỏi/quan tâm đến size (xem rule 0b)
   - Ảnh: dùng markdown ![tên](image_cover) - QUAN TRỌNG để user xem được
   - 1-2 câu ý nghĩa / công dụng — LẤY TỪ product_description (xem rule 0a), KHÔNG bịa từ kiến thức model
3. Chỉ kết luận "không tìm thấy" SAU KHI đã thử fallback (ít nhất 2 cách search,
   xem mục FALLBACK). Khi đó nói rõ và đề xuất hướng khác (đổi tiêu chí, gợi ý
   chat nhân viên, hoặc web_search nếu là sản phẩm shop không bán).
4. Trả lời bằng tiếng Việt, giọng thân thiện, xưng "shop" - gọi khách là "bạn".
5. Không bịa thông tin. Nếu DB không có field nào đó (vd: số hạt theo size,
   giấy chứng chỉ), hãy nói "shop sẽ kiểm tra lại và phản hồi sau, hoặc bạn
   inbox trực tiếp Shopee để được nhân viên hỗ trợ".

6. TUYỆT ĐỐI KHÔNG để LỖI/THUẬT NGỮ NỘI BỘ lọt ra câu trả lời cho khách:
   - KHÔNG bao giờ nhắc "product_id", "id", "tool", "nhầm lẫn id", mã lỗi, JSON...
   - Nếu một tool báo lỗi (vd get_product_detail_tool trả internal_error) → ĐỪNG xin lỗi
     khách về lỗi đó. Hãy TỰ KHẮC PHỤC: gọi keyword_search_tool(query=TÊN sản phẩm đang
     nói tới) để lấy đúng sản phẩm rồi trả lời bình thường. Thử lại 2-3 lần nếu cần.
   - Khi khách hỏi Ý NGHĨA / CÔNG DỤNG / PHONG THỦY của sản phẩm VỪA hiển thị ở lượt
     trước → DÙNG NGAY "product_description" đã có trong hội thoại (theo quy tắc 0a),
     KHÔNG cần gọi lại get_product_detail_tool.
   - CHỈ khi đã thử lại nhiều lần mà thật sự không có dữ liệu → nói GỌN, thân thiện:
     "Dạ shop kiểm tra lại thông tin sản phẩm này rồi báo bạn ngay nhé ạ" — KHÔNG nêu
     lý do kỹ thuật.

7. KHÔNG BAO GIỜ nói "shop không có / không bán / chưa kinh doanh X" khi CHƯA gọi
   tool search. Khách hỏi "shop có bán X không / có X không / bên bạn có X":
   - BẮT BUỘC gọi keyword_search_tool(query=X) (và/hoặc semantic/filter) TRƯỚC.
   - Điều này áp dụng KỂ CẢ khi X nghe KHÔNG giống đồ phong thủy (vd "dầu gió", "tinh
     dầu", "than xông", "nước lau", "miếng dán"...) — shop bán nhiều loại, RẤT có thể có
     trong DB. Đừng tự đoán shop không bán.
   - CHỈ khi tool trả về RỖNG hoặc không có mẫu khớp rõ → mới nói shop chưa có, rồi gợi ý
     sản phẩm liên quan / hỏi nhu cầu khác.
"""


_FINETUNE_NOTE = (
    "\n\n━━━ CHẾ ĐỘ MODEL FINETUNE (ĐANG BẬT) ━━━\n"
    "- Ảnh khách → model finetune nhận diện + DB đã tiêm sẵn (kết quả tool): name, colors, "
    "product_size, compatible_elements, price_range, in_stock, quantity_min/max, image_cover.\n"
    "- Giá/tồn kho CHỈ lấy từ field DB trong kết quả tiêm sẵn (in_stock, quantity_*, price_range) "
    "— model finetune KHÔNG có các field này; đừng bịa.\n"
    "- ƯU TIÊN #1: TRẢ LỜI ĐÚNG CÂU HỎI TEXT của khách trước (còn hàng? giá? size? mệnh?). "
    "Câu 1–2 phải chốt YES/NO hoặc số liệu; SAU ĐÓ mới xác nhận ngắn tên SP + ảnh nếu cần. "
    "CẤM viết đoạn giới thiệu/marketing dài (ý nghĩa đá, ngũ hành, 'lá bùa'...) khi khách "
    "CHỈ hỏi tồn kho/giá/size — trừ khi khách hỏi ý nghĩa.\n"
    " · 'còn hàng không / còn bao nhiêu' → đọc NGUYÊN field stock_display: 'còn nhiều hàng' / "
    "'còn N sản phẩm (sắp hết)' (khi ≤10) / 'hiện hết hàng'. Kèm giá nếu hữu ích.\n"
    " · TUYỆT ĐỐI không đọc số kho thô nếu quá lớn (vd 939235) — đã gói sẵn trong stock_display.\n"
    "- ĐỪNG search lại SP TRONG ẢNH (đã nhận diện). Chỉ search khi hỏi SP/biến thể KHÁC.\n"
    "- Câu chữ không ảnh: semantic/filter/keyword như bình thường.\n"
)


def agent_node(state: MessagesState) -> dict:
    prompt = KB_SYSTEM_PROMPT + (_FINETUNE_NOTE if USE_FINETUNE else "")
    llm = make_llm_with_tools(TOOLS, temperature=0.3)
    response = llm.invoke([SystemMessage(content=prompt)] + list(state["messages"]))
    return {"messages": [response]}


def should_continue(state: MessagesState) -> str:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END


_graph = None


def build_graph():
    g = StateGraph(MessagesState)
    g.add_node("agent", agent_node)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# Trả lời CỐ ĐỊNH khi khách gửi ảnh KHÔNG phải sản phẩm của shop (người/thú/xe/...).
IRRELEVANT_IMAGE_REPLY = (
    "Dạ ảnh bạn gửi shop chưa nhận ra là sản phẩm của shop ạ. Shop chỉ hỗ trợ tư vấn "
    "các sản phẩm phong thủy của Vạn An Group (vòng tay, trầm hương, lư xông, đá phong "
    "thủy, treo xe...). Bạn gửi giúp shop ảnh sản phẩm cần tư vấn để shop hỗ trợ nhé!"
)

# Model finetune KHÔNG gọi được (ngrok chết / Colab ngắt / timeout). Đây là SỰ CỐ KỸ
# THUẬT của shop — KHÔNG được nói với khách là ảnh của họ "không phải sản phẩm shop",
# vì rất có thể đó ĐÚNG là sản phẩm shop bán. Nói thật, đừng đổ lỗi cho khách.
IMAGE_SERVICE_DOWN_REPLY = (
    "Dạ hệ thống nhận diện ảnh của shop đang tạm trục trặc, shop chưa xem được ảnh bạn "
    "gửi ạ. Bạn thử gửi lại sau ít phút giúp shop nhé — hoặc cho shop biết TÊN sản phẩm "
    "(hay mô tả giúp shop: loại gì, chất liệu, màu nào) thì shop tư vấn ngay được ạ!"
)


def _latest_user_text(messages: list[BaseMessage]) -> str:
    """Text câu hỏi khách ở HumanMessage mới nhất (bỏ phần ảnh)."""
    for m in reversed(messages):
        if not isinstance(m, HumanMessage):
            continue
        c = m.content
        if isinstance(c, str):
            return c.strip()
        if isinstance(c, list):
            parts = [
                (p.get("text") or "").strip()
                for p in c
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            return " ".join(x for x in parts if x).strip()
    return ""


def _stock_phrase(in_stock: bool, qty) -> str:
    """Câu tồn kho tính sẵn để LLM đọc NGUYÊN (đừng để LLM tự đoán số to/nhỏ).
      • hết hàng → "hết hàng"
      • còn ≤10 → "còn N sản phẩm (sắp hết)"
      • còn nhiều → "còn nhiều hàng"
    Dùng quantity_max (cột admin nạp tồn kho + in_stock=quantity_max>0)."""
    if not in_stock:
        return "hết hàng"
    if not qty or qty <= 0:
        return "còn hàng"
    if qty <= 10:
        return f"còn {qty} sản phẩm (sắp hết)"
    return "còn nhiều hàng"


def _slim_product_for_seed(p: dict, desc_chars: int = 420) -> dict:
    """Seed đủ field trả lời thực dụng; giữ product_description vừa đủ để trả size/quy cách."""
    out = {
        "product_id": p.get("product_id"),
        "name": p.get("name"),
        "category": p.get("category"),
        "material": p.get("material"),
        "compatible_elements": p.get("compatible_elements"),
        "colors": p.get("colors"),
        "product_size": p.get("product_size"),
        "price_range": p.get("price_range"),
        "in_stock": p.get("in_stock"),
        "quantity_min": p.get("quantity_min"),
        "quantity_max": p.get("quantity_max"),
        "image_cover": p.get("image_cover"),
        "warranty": p.get("warranty"),
        # tóm tắt tồn kho dễ đọc (tính sẵn — LLM đọc NGUYÊN stock_display, không đọc số kho thô)
        "stock_status": "con_hang" if p.get("in_stock") else "het_hang",
        "stock_display": _stock_phrase(bool(p.get("in_stock")), p.get("quantity_max")),
    }
    desc = p.get("product_description") or ""
    if desc:
        out["product_description_preview"] = (
            (desc[:desc_chars] + "…") if len(desc) > desc_chars else desc
        )
        out["note_description"] = (
            "Dùng preview để trả chất liệu/kích thước/quy cách nếu có trong text. "
            "Chỉ gọi get_product_detail_tool khi preview thiếu ý khách hỏi "
            "(vd size/quy cách không có trong product_size lẫn preview)."
        )
    return out


def _question_aspects_hint(user_q: str) -> list[str]:
    """Gợi ý các khía cạnh khách có thể đang hỏi (để seed checklist; model vẫn tự reasoning)."""
    q = (user_q or "").lower()
    aspects: list[str] = []
    pairs = [
        (["chất liệu", "lam tu", "làm từ", "chat lieu", "vật liệu", "vat lieu", "chất liệu gì"],
         "chất_liệu→material"),
        (["kích thước", "kich thuoc", "size", "số đo", "so do", "quy cách", "quy cach",
          "đường kính", "chiều cao", "rộng", "cm", "mm", "bao nhiêu cm", "to nhỏ"],
         "kích_thước/quy_cách→product_size+product_description_preview"),
        (["màu", "mau ", "color"], "màu→colors"),
        (["giá", "gia ", "bao nhiêu tiền", "price"], "giá→price_range"),
        (["còn hàng", "con hang", "hết hàng", "tồn", "còn không"], "tồn_kho→stock_display"),
        (["mệnh", "hợp", "kỵ", "năm sinh", "tuổi"], "mệnh→compatible_elements"),
        (["bảo hành", "bao hanh", "thay dây"], "bảo_hành→warranty"),
    ]
    for keys, label in pairs:
        if any(k in q for k in keys):
            aspects.append(label)
    if not aspects and q.strip():
        aspects.append("trả_lời_đủ_mọi_ý_trong_user_question_dùng_metadata_SP")
    return aspects


def _seed_messages(messages: list[BaseMessage], products: list[dict]) -> list[BaseMessage]:
    """TIÊM sản phẩm THẬT (đã nhận diện) vào hội thoại dưới dạng tool-result, NGAY
    TRƯỚC khi model trả lời.

    Lý do: model đa phương thức thường KHÔNG chịu gọi tool khi 'nhìn thấy' ảnh →
    bịa hoặc hỏi lại mà không nêu được sản phẩm. Tiêm sẵn sản phẩm thật giúp model
    luôn mô tả/đối chiếu mệnh/recommend đúng, và tên sản phẩm cũng đi vào câu trả
    lời (→ lưu vào history cho các lượt sau anchor được)."""
    if not products:
        return messages
    user_q = _latest_user_text(messages)
    aspects = _question_aspects_hint(user_q)
    # Câu hỏi nhiều ý (chất liệu+size…) → cho preview mô tả dài hơn để lấy quy cách
    multi = len(aspects) >= 2 or ("chất_liệu" in " ".join(aspects) and "kích_thước" in " ".join(aspects))
    desc_chars = 520 if multi else 420
    slim = [_slim_product_for_seed(p, desc_chars=desc_chars) for p in products]
    call_id = "img_identify"
    seed_ai = AIMessage(content="", tool_calls=[
        {"name": _SEED_TOOL_NAME, "args": {}, "id": call_id},
    ])
    seed_tool = ToolMessage(
        content=json.dumps({
            "matched": True,
            "num_images": len(slim),
            "candidates": slim,
            "per_image": [{"image_index": i + 1, "matched": True, "best_product": p}
                          for i, p in enumerate(slim)],
            "user_question": user_q,
            "question_aspects_hint": aspects,
            "priority": (
                "BẮT BUỘC SUY LUẬN: user_question có thể có NHIỀU Ý trong 1 câu. "
                "Tách TỪNG ý rồi trả lời ĐỦ TẤT CẢ trong MỘT reply (gạch ý hoặc đánh số). "
                f"Gợi ý các ý phát hiện: {aspects}. "
                "Metadata SP đã có trong candidates — map: "
                "chất liệu→material; kích thước/quy cách→product_size + product_description_preview; "
                "màu→colors; giá→price_range; còn hàng→stock_display. "
                "Ví dụ 'chất liệu gì kích thước thế nào' → (1) material (2) product_size/mô tả. "
                "CẤM chỉ trả 1 ý. CẤM 'shop sẽ kiểm tra size/chất liệu' nếu field tương ứng "
                "đã có dữ liệu. CẤM marketing/ý nghĩa đá trừ khi user hỏi."
            ),
            "note": (
                "Sản phẩm THẬT: finetune/SigLIP nhận diện + Postgres (đủ metadata). "
                "CHỈ dùng field candidates; KHÔNG đọc giá từ ảnh, KHÔNG bịa. "
                "product_size rỗng mà khách hỏi size → đọc product_description_preview; "
                "vẫn thiếu → get_product_detail_tool(product_id) rồi trả lời đủ các ý."
            ),
        }, ensure_ascii=False),
        tool_call_id=call_id,
        name=_SEED_TOOL_NAME,
    )
    return list(messages) + [seed_ai, seed_tool]


def run(messages: list[BaseMessage]) -> dict:
    """Public entrypoint used by graph.py."""
    log.info("ENTER knowledge_base_agent (%d msgs)", len(messages))

    # Lưới an toàn: tin nhắn có ảnh nhưng ý định là khiếu nại/handoff → KHÔNG gọi FT.
    # Supervisor thường đã route order_support; nếu vẫn vào KB thì skip identify.
    try:
        from image_turn_intent import (
            classify_image_turn_intent,
            message_has_image,
        )
        if message_has_image(messages):
            decision = classify_image_turn_intent(messages, default="identify")
            if decision.get("intent") == "escalate":
                log.info(
                    "EXIT knowledge_base_agent | skip FT (intent=escalate) → "
                    "nhường order_support/handoff | reason=%s",
                    (decision.get("reason") or "")[:100],
                )
                # Không identify; để ReAct KB trả lời ngắn hoặc caller đã handoff.
                # Trả lời an toàn: hướng khách chờ shop / không nhận diện catalog.
                msg = (
                    "Dạ em đã ghi nhận yêu cầu (kèm ảnh) và sẽ chuyển cho chủ shop xử lý "
                    "trực tiếp ạ. Bạn vui lòng chờ shop phản hồi tại đây trong thời gian "
                    "sớm nhất nhé ạ."
                )
                # Gọi escalate tool để đồng bộ ticket + pending_admin (graph bắt tool name).
                try:
                    from order_support_agent import escalate_to_human_tool
                    esc = escalate_to_human_tool.invoke({
                        "reason": "complaint",
                        "user_summary": (decision.get("reason") or "Khách gửi ảnh + yêu cầu cần shop")[:200],
                    })
                    import json as _json
                    esc_d = _json.loads(esc) if isinstance(esc, str) else {}
                    if esc_d.get("message_for_user"):
                        msg = esc_d["message_for_user"]
                except Exception as ex:
                    log.warning("KB skip-FT escalate helper lỗi: %s", ex)
                # Gắn tool_calls để graph.py detect escalate → pending_admin.
                return {
                    "final_response": msg,
                    "messages": list(messages) + [AIMessage(
                        content=msg,
                        tool_calls=[{
                            "name": "escalate_to_human_tool",
                            "args": {
                                "reason": "complaint",
                                "user_summary": (decision.get("reason") or "")[:200],
                            },
                            "id": "kb_skip_ft_escalate",
                        }],
                    )],
                    "tools_called": ["escalate_to_human_tool"],
                }
    except Exception as ex:
        log.warning("KB image-intent guard lỗi (tiếp tục identify): %s", ex)

    # Ảnh + intent identify → nhận diện SP trước khi LLM trả lời.
    # FT (nếu bật) trước; không biết / lỗi → SigLIP vector search (identify_customer_images).
    info = identify_customer_images(messages)

    # Chỉ báo sập khi FT lỗi VÀ SigLIP fallback cũng không cứu được (api_error còn).
    if info.get("api_error"):
        log.error(
            "EXIT knowledge_base_agent | FINETUNE lỗi và SigLIP không map được (%s) "
            "FINETUNE_API_URL=%s",
            info["api_error"], FINETUNE_API_URL,
        )
        return {
            "final_response": IMAGE_SERVICE_DOWN_REPLY,
            "messages": list(messages) + [AIMessage(content=IMAGE_SERVICE_DOWN_REPLY)],
            "tools_called": [],
        }

    # Ảnh KHÔNG phải sản phẩm shop (người/thú/xe/...) → từ chối khéo, không vào agent.
    if info["has_image"] and not info["any_product_like"]:
        log.info("EXIT knowledge_base_agent | ảnh không liên quan sản phẩm shop → từ chối khéo")
        return {
            "final_response": IRRELEVANT_IMAGE_REPLY,
            "messages": list(messages) + [AIMessage(content=IRRELEVANT_IMAGE_REPLY)],
            "tools_called": [],
        }
    # Tiêm sản phẩm thật (nếu nhận diện được) vào hội thoại.
    messages = _seed_messages(messages, info["products"])
    if info.get("has_image") and info["products"]:
        progress.emit("answering", "Đang soạn câu trả lời cho bạn...")

    # Đưa ảnh khách (nếu có) vào contextvar để image_search_tool truy cập được.
    token = _QUERY_IMAGE.set(_extract_query_images_bytes(messages))
    try:
        result = get_graph().invoke(
            {"messages": messages},
            config={"callbacks": [_callback]},
        )
    finally:
        _QUERY_IMAGE.reset(token)
    final = result["messages"][-1].content
    tools_called = sorted({
        tc["name"]
        for m in result["messages"]
        for tc in getattr(m, "tool_calls", []) or []
    })
    log.info("EXIT knowledge_base_agent | tools=%s | reply=%d chars",
             tools_called, len(final) if isinstance(final, str) else 0)
    return {
        "final_response": final,
        "messages": result["messages"],
        "tools_called": tools_called,
    }
