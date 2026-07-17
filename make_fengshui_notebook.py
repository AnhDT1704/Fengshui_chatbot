"""
make_fengshui_notebook.py – Sinh notebook Colab để finetune Qwen3-4B trên dataset
phong thủy, chạy được CẢ HAI biến thể A (đáp án thẳng) và B (có <think>).

Chạy:  python make_fengshui_notebook.py
       → data/qwen3_fengshui_finetune.ipynb
"""
import json
from pathlib import Path


def _lines(src: str) -> list[str]:
    """Chuẩn .ipynb: mỗi phần tử của `source` phải KẾT THÚC BẰNG '\\n' (trừ dòng cuối).
    Thiếu '\\n' thì Jupyter/Colab nối hết thành MỘT dòng dài — rất khó đọc."""
    lines = src.strip().split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def md(src):   return {"cell_type": "markdown", "metadata": {}, "source": _lines(src)}
def code(src): return {"cell_type": "code", "metadata": {}, "execution_count": None,
                       "outputs": [], "source": _lines(src)}


CELLS = [
md("""
# Finetune Qwen3-4B — TRI THỨC PHONG THỦY (Vạn An Group)

Dạy model 4 kỹ năng:
1. **Năm sinh** → Can Chi → Nạp âm → Mệnh → màu hợp/kỵ
2. **Can Chi đầy đủ** (vd "Canh Ngọ") → mệnh
3. **Chỉ có con giáp** (vd "tuổi Ngọ") → **HỎI LẠI năm sinh** (1 con giáp có 5 mệnh!)
4. **Cổ tay (cm)** → size hạt → số hạt → cung Sinh-Lão-Bệnh-Tử

## Thí nghiệm: model HỌC QUY LUẬT hay HỌC VẸT?

Chạy notebook **2 lần**, chỉ đổi biến `VARIANT`:

| | Dữ liệu | Model nền |
|---|---|---|
| **A** | đáp án thẳng | Qwen3-4B-Instruct |
| **B** | có `<think>` (viết ra phép tính) | Qwen3-4B-Instruct *(giống A)* |

Giữ **cùng model nền**, chỉ đổi dữ liệu → so sánh SẠCH, đo đúng tác động của suy luận.

Đánh giá trên **2 tập test**:
- `test.jsonl` — năm/cổ tay **chưa từng thấy**, nhưng trong vùng đã train → **nội suy**
- `test_extrapolate.jsonl` — năm **1840-1899 + 2101-2160**, ngoài hẳn vùng train → **NGOẠI SUY**

> Nếu model thật sự học được `(năm − 1924) % 60` → nó trả lời đúng cả năm 2137.
> Nếu chỉ học vẹt → sập trên tập ngoại suy.

**Trước khi chạy:** Runtime → **T4 GPU**; kéo `dataset_fengshui.zip` **và** `dataset_fengshui_cot.zip` vào Files (📁).
"""),

md("## 1. Cài Unsloth"),
code("""
%%capture
import os, re
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth
else:
    import torch; v = re.match(r'[\\d]{1,}\\.[\\d]{1,}', str(torch.__version__)).group(0)
    xformers = 'xformers==' + {'2.10':'0.0.34','2.9':'0.0.33.post1','2.8':'0.0.32.post2'}.get(v, "0.0.34")
    !pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer
    !pip install --no-deps unsloth_zoo bitsandbytes accelerate {xformers} peft trl triton unsloth
    !pip install --no-deps --upgrade "torchao>=0.16.0"
!pip install transformers==4.56.2
!pip install --no-deps trl==0.22.2
"""),

md("""
## 2. Cấu hình — ĐỔI `VARIANT` để chạy biến thể khác

Chạy lần 1 với `VARIANT = "A"`, lần 2 với `VARIANT = "B"`, rồi so bảng kết quả cuối.
"""),
code("""
# ════════════════════════════════════════════════════════════════
VARIANT = "A"        # <<<<<<  "A" = đáp án thẳng  |  "B" = có <think>
# ════════════════════════════════════════════════════════════════

DATA_ZIP = "dataset_fengshui.zip" if VARIANT == "A" else "dataset_fengshui_cot.zip"
DATA_DIR = "dataset_fengshui"     if VARIANT == "A" else "dataset_fengshui_cot"

# CÙNG model nền cho cả 2 biến thể → so sánh sạch, chỉ đổi DỮ LIỆU.
MODEL_NAME = "unsloth/Qwen3-4B-Instruct-2507"

# Biến thể B có <think> nên đáp án dài gấp ~3 lần → cần seq dài hơn.
MAX_SEQ_LEN  = 1024 if VARIANT == "A" else 2048
MAX_NEW_TOK  = 320  if VARIANT == "A" else 900   # lúc sinh câu trả lời để chấm điểm

# CHỐNG OOM: B seq dài gấp đôi → hạ batch xuống 1, bù bằng gradient accumulation.
# Batch hiệu dụng vẫn = 8 ở cả hai biến thể → so sánh A/B vẫn công bằng.
BATCH  = 2 if VARIANT == "A" else 1
GRAD_ACC = 4 if VARIANT == "A" else 8

!unzip -q -o {DATA_ZIP} -d /content/

# Mount Drive NGAY TỪ ĐẦU: Colab free hay ngắt giữa chừng. Checkpoint phải nằm trên
# Drive từ epoch đầu tiên, không phải chỉ lưu lúc train xong — đứt là mất trắng.
from google.colab import drive
drive.mount('/content/drive')
import os
OUT_DIR = f"/content/drive/MyDrive/qwen3_fengshui_{VARIANT}"
os.makedirs(OUT_DIR, exist_ok=True)

print("VARIANT =", VARIANT, "| data:", DATA_DIR, "| max_seq_len:", MAX_SEQ_LEN)
print("Checkpoint sẽ lưu vào:", OUT_DIR)
!ls /content/{DATA_DIR}
"""),

md("## 3. Nạp Qwen3-4B + gắn LoRA"),
code("""
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = MODEL_NAME,
    max_seq_length = MAX_SEQ_LEN,
    load_in_4bit   = True,
    full_finetuning= False,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 32,
    target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)
"""),

md("## 4. Nạp dataset (JSONL của mình) + áp chat template Qwen3"),
code("""
import json
from datasets import Dataset
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(tokenizer, chat_template="qwen3-instruct")

def load_jsonl(name):
    rows = [json.loads(l) for l in open(f"/content/{DATA_DIR}/{name}.jsonl", encoding="utf-8")]
    return Dataset.from_list([{"conversations": r["messages"]} for r in rows])

train_ds = load_jsonl("train")
valid_ds = load_jsonl("valid")

def to_text(ex):
    return {"text": [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
                     for c in ex["conversations"]]}

train_ds = train_ds.map(to_text, batched=True)
valid_ds = valid_ds.map(to_text, batched=True)
print("train:", len(train_ds), "| valid:", len(valid_ds))
print()
print(train_ds[0]["text"][:700])
"""),

md("""
## 5. Hàm CHẤM ĐIỂM — trái tim của thí nghiệm

Chấm **từng trường** so với đáp án (đáp án lấy từ code chatbot → đúng 100%).
Chấm riêng 2 nhiệm vụ và cả hành vi **"biết hỏi lại"**.
"""),
code("""
import json, re, torch
from transformers import TextStreamer

MENH_FIELDS = ["can_chi", "napam", "element", "generating_element",
               "controlling_element", "lucky_colors", "unlucky_colors"]
SIZE_FIELDS = ["bead_size_li", "bead_count", "length_cm", "slack_cm", "fengshui"]

def parse_json(txt):
    # Biến thể B có <think>...</think> trước JSON → bỏ đi rồi mới parse.
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S)
    s, e = txt.find("{"), txt.rfind("}")
    try:
        return json.loads(txt[s:e+1]) if s != -1 and e != -1 else {}
    except Exception:
        return {}

def eq(a, b):
    if isinstance(b, list): return set(map(str, a or [])) == set(map(str, b))
    return str(a) == str(b)

@torch.no_grad()
def evaluate(path, n=None, show=0):
    FastLanguageModel.for_inference(model)
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    if n: rows = rows[:n]

    hits, tot = {}, {}
    jsonok = 0
    askback_ok = askback_tot = 0

    for k, r in enumerate(rows):
        msgs = r["messages"][:2]                       # system + user
        gold = parse_json(r["messages"][2]["content"])
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        out  = model.generate(**tokenizer(text, return_tensors="pt").to("cuda"),
                              max_new_tokens=MAX_NEW_TOK, do_sample=False)
        gen  = tokenizer.decode(out[0][len(tokenizer(text)["input_ids"]):], skip_special_tokens=True)
        pred = parse_json(gen)
        if pred: jsonok += 1

        if k < show:
            print("Q:", msgs[1]["content"]); print("A:", gen[:400]); print("-"*50)

        # Hành vi "biết hỏi lại" khi thiếu dữ kiện (chỉ có con giáp)
        if gold.get("need_more_info"):
            askback_tot += 1
            askback_ok += bool(pred.get("need_more_info"))
            continue

        fields = SIZE_FIELDS if gold.get("task") == "size" else MENH_FIELDS
        for f in fields:
            if f not in gold: continue
            tot[f] = tot.get(f, 0) + 1
            hits[f] = hits.get(f, 0) + eq(pred.get(f), gold[f])

    FastLanguageModel.for_training(model)
    res = {f: hits.get(f, 0) / tot[f] * 100 for f in tot}
    res["_json_hop_le"] = jsonok / len(rows) * 100
    if askback_tot:
        res["_biet_hoi_lai"] = askback_ok / askback_tot * 100
    res["_n"] = len(rows)
    return res

def show_table(title, res):
    print(f"\\n===== {title}  (n={res['_n']}) =====")
    for f, v in res.items():
        if f.startswith("_"): continue
        print(f"   {f:22s}: {v:5.1f}%")
    print(f"   {'JSON hợp lệ':22s}: {res['_json_hop_le']:5.1f}%")
    if "_biet_hoi_lai" in res:
        print(f"   {'BIẾT HỎI LẠI':22s}: {res['_biet_hoi_lai']:5.1f}%")
"""),

md("""
## 6. Kiểm tra VRAM trước khi train (chống OOM)
"""),
code("""
import torch
g = torch.cuda.get_device_properties(0)
tot  = g.total_memory / 1e9
used = torch.cuda.memory_allocated() / 1e9
print(f"GPU: {g.name} | VRAM tổng {tot:.1f} GB | model đang chiếm {used:.1f} GB")
print(f"Còn trống ~{tot - used:.1f} GB cho activation")
print(f"Cấu hình: seq={MAX_SEQ_LEN} batch={BATCH} grad_acc={GRAD_ACC} (batch hiệu dụng {BATCH*GRAD_ACC})")
print()
if tot - used < 6:
    print("⚠ Còn ít VRAM. Nếu OOM khi train: hạ MAX_SEQ_LEN hoặc đặt BATCH=1 ở cell 2.")
else:
    print("✓ Dư VRAM, train được.")
"""),

md("""
## 7. Train — có ĐO ACCURACY mỗi epoch + EARLY STOPPING

- **Metrics mỗi epoch**: `eval_loss` + **độ chính xác từng trường** (chạy model trên
  một ít mẫu valid → biết ngay nó học được gì, không phải đoán qua loss).
- **Early stopping**: `eval_loss` 2 epoch liền không cải thiện → dừng, giữ bản tốt nhất.
  Tránh train thừa (tốn giờ Colab) và tránh overfit.
"""),
code("""
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback, EarlyStoppingCallback
from unsloth.chat_templates import train_on_responses_only

class FieldAccuracyCallback(TrainerCallback):
    \"\"\"Sau mỗi epoch: sinh câu trả lời trên N mẫu valid, chấm ĐỘ CHÍNH XÁC TỪNG TRƯỜNG.

    Vì sao cần: eval_loss giảm KHÔNG có nghĩa model trả lời đúng. Loss chỉ đo model
    đoán token kế tiếp giỏi tới đâu; nó có thể giảm đẹp mà 'Nạp âm' vẫn sai bét.
    Đây là thứ ta thật sự quan tâm.
    \"\"\"
    def __init__(self, n=24):
        self.n = n
        self.history = []

    def on_evaluate(self, args, state, control, **kw):
        ep = int(round(state.epoch or 0))
        res = evaluate(f"/content/{DATA_DIR}/valid.jsonl", n=self.n)
        self.history.append((ep, res))
        keys = [k for k in ("element", "napam", "can_chi", "bead_count", "fengshui") if k in res]
        line = " | ".join(f"{k}={res[k]:.0f}%" for k in keys)
        extra = f" | JSON={res['_json_hop_le']:.0f}%"
        if "_biet_hoi_lai" in res:
            extra += f" | hỏi-lại={res['_biet_hoi_lai']:.0f}%"
        print(f"\\n  [epoch {ep}] {line}{extra}\\n")

acc_cb = FieldAccuracyCallback(n=24)

trainer = SFTTrainer(
    model = model, tokenizer = tokenizer,
    train_dataset = train_ds, eval_dataset = valid_ds,
    callbacks = [
        acc_cb,
        # Dừng khi eval_loss 2 epoch liền không cải thiện.
        EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.001),
    ],
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = BATCH,
        gradient_accumulation_steps = GRAD_ACC,
        warmup_steps = 5,
        num_train_epochs = 10,         # TRẦN — early stopping sẽ tự dừng sớm
        learning_rate = 2e-4,
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none",

        # Bắt buộc để early stopping + giữ bản tốt nhất hoạt động:
        eval_strategy = "epoch",
        save_strategy = "epoch",
        load_best_model_at_end = True,
        metric_for_best_model = "eval_loss",
        greater_is_better = False,

        # ── CHỐNG MẤT TIẾN ĐỘ khi Colab ngắt (đã xảy ra nhiều lần) ──
        output_dir = OUT_DIR,          # lưu thẳng lên DRIVE, không phải ổ tạm
        save_total_limit = 2,
    ),
)
# Chỉ tính loss trên phần TRẢ LỜI, bỏ qua phần câu hỏi -> chính xác hơn.
trainer = train_on_responses_only(trainer)

# Colab đứt giữa chừng? Chạy lại từ cell 1 là TỰ RESUME từ epoch dở, không mất gì.
import glob
_ck = glob.glob(f"{OUT_DIR}/checkpoint-*")
print("Có checkpoint cũ để resume:", bool(_ck), _ck)
trainer_stats = trainer.train(resume_from_checkpoint=bool(_ck))
"""),

md("### Diễn biến accuracy qua các epoch"),
code("""
print(f"{'epoch':>6} | " + " | ".join(f"{k:>10}" for k in
      ("element", "napam", "bead_count", "fengshui", "JSON")))
print("-" * 66)
for ep, r in acc_cb.history:
    print(f"{ep:>6} | " + " | ".join(
        f"{r.get(k, 0):9.0f}%" for k in ("element", "napam", "bead_count", "fengshui")) +
        f" | {r['_json_hop_le']:9.0f}%")
"""),

md("""
## 8. KẾT QUẢ — bảng trả lời câu hỏi nghiên cứu

Đây là con số bạn đưa vào báo cáo.
"""),
code("""
r_in  = evaluate(f"/content/{DATA_DIR}/test.jsonl", show=2)
r_out = evaluate(f"/content/{DATA_DIR}/test_extrapolate.jsonl", n=120)

show_table(f"[{VARIANT}] TRONG VÙNG  (năm chưa thấy, 1900-2100)", r_in)
show_table(f"[{VARIANT}] NGOÀI VÙNG  (năm 1840-1899 + 2101-2160)", r_out)

f = "element"
print("\\n" + "="*62)
print(f"KẾT LUẬN (biến thể {VARIANT}) — độ chính xác MỆNH:")
print(f"   trong vùng : {r_in.get(f,0):5.1f}%")
print(f"   ngoài vùng : {r_out.get(f,0):5.1f}%")
d = r_in.get(f,0) - r_out.get(f,0)
print(f"   chênh lệch : {d:5.1f} điểm")
print()
print("   Chênh lệch NHỎ  -> model học được QUY LUẬT (năm-1924)%60")
print("   Chênh lệch LỚN  -> model chỉ HỌC VẸT các năm đã thấy")
print("="*62)
"""),

md("""
## 9. Lưu adapter + KẾT QUẢ lên Drive

Drive đã mount ở mục 2, checkpoint cũng đã tự lưu mỗi epoch. Cell này lưu **bản
adapter cuối (tốt nhất)** + **file kết quả** để bạn đối chiếu A vs B sau này.
"""),
code("""
import json

# Adapter bản tốt nhất (load_best_model_at_end đã nạp lại checkpoint có eval_loss thấp nhất)
model.save_pretrained(f"{OUT_DIR}/final")
tokenizer.save_pretrained(f"{OUT_DIR}/final")

json.dump({
    "variant":          VARIANT,
    "model":            MODEL_NAME,
    "train_loss":       trainer_stats.metrics.get("train_loss"),
    "test_in_range":    r_in,     # năm chưa thấy, TRONG vùng 1900-2100
    "test_extrapolate": r_out,    # năm NGOÀI vùng train
}, open(f"{OUT_DIR}/results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

print("Đã lưu adapter :", f"{OUT_DIR}/final")
print("Đã lưu kết quả :", f"{OUT_DIR}/results.json")
"""),

md("""
## 10. So sánh A vs B  *(chạy sau khi đã train XONG CẢ HAI)*

Đọc `results.json` của cả 2 biến thể → in bảng cuối cùng cho báo cáo.
"""),
code("""
import json, os

rows = {}
for v in ("A", "B"):
    p = f"/content/drive/MyDrive/qwen3_fengshui_{v}/results.json"
    if os.path.exists(p):
        rows[v] = json.load(open(p, encoding="utf-8"))
    else:
        print(f"Chưa có kết quả biến thể {v} — hãy train nó trước.")

if len(rows) == 2:
    print(f"{'':28s} {'A (đáp án thẳng)':>18s} {'B (có <think>)':>18s}")
    print("-" * 68)
    for label, key in (("MỆNH - trong vùng", "test_in_range"),
                       ("MỆNH - NGOÀI vùng", "test_extrapolate")):
        a = rows["A"][key].get("element", 0)
        b = rows["B"][key].get("element", 0)
        print(f"{label:28s} {a:17.1f}% {b:17.1f}%")
    print("-" * 68)
    for v in ("A", "B"):
        d = rows[v]["test_in_range"].get("element", 0) - rows[v]["test_extrapolate"].get("element", 0)
        verdict = "HỌC ĐƯỢC QUY LUẬT" if d < 15 else "chỉ HỌC VẸT"
        print(f"  {v}: chênh lệch trong/ngoài vùng = {d:5.1f} điểm  ->  {verdict}")
"""),
]

nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = Path("data/qwen3_fengshui_finetune.ipynb")
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Đã tạo:", out, f"({len(CELLS)} cells)")
