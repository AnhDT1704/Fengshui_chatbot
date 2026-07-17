"""
serve_finetune.py – Serve model VLM đã finetune (Qwen2.5-VL-7B + LoRA/Unsloth) NGAY
TRÊN MÁY CÓ GPU, thay cho Colab + ngrok.

Vì sao bỏ Colab/ngrok: tunnel ngrok chết liên tục (phiên Colab ngắt, URL đổi mỗi lần
chạy lại). Khi nó chết, chatbot không nhận diện được ảnh. Chạy local → endpoint cố định.

CHẠY (trên máy có GPU ≥12GB, khuyến nghị Linux hoặc WSL2 — Unsloth hỗ trợ Windows kém):

    pip install unsloth fastapi "uvicorn[standard]" python-multipart pillow
    python serve_finetune.py --model-path /duong/dan/checkpoint-392

    # đổi cổng nếu 8001 bận (8000 đã bị chatbot Docker chiếm)
    python serve_finetune.py --model-path ... --port 8001

Kiểm tra:
    curl http://localhost:8001/health
    curl -X POST -F "file=@anh.jpg" http://localhost:8001/predict

Trỏ chatbot vào (docker-compose.yml, service `chatbot`):
    # Docker Desktop (Windows/Mac): container gọi host qua host.docker.internal
    FINETUNE_API_URL: "http://host.docker.internal:8001"
    # Linux: dùng IP máy host, vd "http://172.17.0.1:8001"
rồi:  docker compose up -d --force-recreate chatbot
"""

from __future__ import annotations

import argparse
import io
import json
import re

import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image

# Phải khớp CHÍNH XÁC system message lúc train (qwen7b_unsloth_fengshui.ipynb, cell
# nạp dataset). Lệch một chữ là model trả lời khác đi.
SYSTEM_MESSAGE = (
    "Bạn là mô hình thị giác-ngôn ngữ chuyên trích xuất thông tin sản phẩm phong thủy "
    "từ ảnh của shop Vạn An Group. Nhìn ảnh và trả về JSON gồm các trường: name, category, "
    "material, colors, product_size, compatible_elements (Kim/Mộc/Thủy/Hỏa/Thổ). "
    "CHỈ trả về JSON, không giải thích thêm."
)

DEFAULT_PREFIX = "Trích xuất thông tin sản phẩm phong thủy trong ảnh dưới dạng JSON."

# Lúc train, ảnh được resize cạnh dài ≤ IMG_MAX (xem _load_img trong notebook). Phải
# resize Y HỆT lúc serve — vừa khớp phân phối train, vừa giảm mạnh vision token (VRAM).
IMG_MAX = 1024

model = None
tokenizer = None
app = FastAPI(title="Fengshui VLM finetune", version="1.0")


def load_img(raw: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = im.size
    if max(w, h) > IMG_MAX:
        s = IMG_MAX / max(w, h)
        im = im.resize((int(w * s), int(h * s)))
    return im


def parse_result(out: str, full: bool) -> dict:
    """Parse JSON model trả về.

    Model xuất newline THẬT bên trong chuỗi product_description → JSON không hợp lệ,
    json.loads() sẽ chết. Escape ký tự điều khiển trước khi parse mới đọc được.

    full=False: output bị CẮT giữa chừng (max_new_tokens thấp) nên không có dấu } đóng.
    Ta cắt phăng từ '"product_description"' rồi tự đóng ngoặc → vẫn lấy đủ 6 trường đầu.
    """
    s = out.find("{")
    if s == -1:
        return {"raw": out}
    frag = out[s:]

    if not full:
        cut = frag.find('"product_description"')
        if cut != -1:
            frag = frag[:cut].rstrip().rstrip(",") + "}"
        else:
            e = frag.rfind("}")
            frag = frag[:e + 1] if e != -1 else frag.rstrip().rstrip(",") + "}"
    else:
        e = frag.rfind("}")
        if e == -1:
            return {"raw": out}
        frag = frag[:e + 1]

    frag = re.sub(r"[\x00-\x1f]", lambda m: "\\u%04x" % ord(m.group()), frag)
    try:
        return json.loads(frag)
    except Exception:
        return {"raw": out}


# product_description là trường CUỐI trong JSON. Dừng sinh chữ trước nó thì 6 trường
# kia đã xong → nhanh gấp ~10 lần, mà chatbot không mất gì (mô tả lấy từ Postgres).
MAX_TOKENS_FULL = 2048   # đủ để viết trọn product_description (~1400 ký tự)
MAX_TOKENS_FAST = 256    # đủ 6 trường đầu (~250 ký tự), cắt trước description


def run_predict(image: Image.Image, prefix: str, full: bool) -> str:
    msgs = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prefix}]},
    ]
    input_text = tokenizer.apply_chat_template(msgs, add_generation_prompt=True)
    inputs = tokenizer(image, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")
    out = model.generate(
        **inputs,
        max_new_tokens=MAX_TOKENS_FULL if full else MAX_TOKENS_FAST,
        use_cache=True,
        do_sample=False,
    )
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...),
                  prefix: str = Form(DEFAULT_PREFIX),
                  full: str = Form("true")):
    """full=false → bỏ product_description, nhanh ~10x (chatbot lấy mô tả từ Postgres)."""
    is_full = str(full).lower() in ("1", "true", "yes")
    image = load_img(await file.read())
    return {"result": parse_result(run_predict(image, prefix, is_full), is_full)}


def main():
    global model, tokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True,
                    help="Thư mục adapter đã train (vd .../qwen7b_fengshui_outputs/checkpoint-392)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8001,
                    help="Mặc định 8001 — KHÔNG dùng 8000 (chatbot Docker đã chiếm)")
    ap.add_argument("--img-max", type=int, default=IMG_MAX,
                    help="Cạnh dài tối đa của ảnh. Hạ 768 (hoặc 512) nếu OOM trên GPU 12GB")
    args = ap.parse_args()

    globals()["IMG_MAX"] = args.img_max

    from unsloth import FastVisionModel

    print(f"Nạp model từ: {args.model_path}")
    model, tokenizer = FastVisionModel.from_pretrained(args.model_path, load_in_4bit=True)
    FastVisionModel.for_inference(model)

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        used = torch.cuda.memory_allocated() / 1e9
        print(f"GPU: {name} | VRAM {vram:.1f}GB | model chiếm ~{used:.1f}GB")
    else:
        print("CẢNH BÁO: không thấy CUDA — sẽ rất chậm.")

    print(f"\nSẵn sàng: http://{args.host}:{args.port}")
    print(f"  Chatbot (Docker Desktop) trỏ vào: http://host.docker.internal:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
