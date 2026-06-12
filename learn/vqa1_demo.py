"""
Bai 16: Visual Q&A — hoi dap ve noi dung hinh anh bang LLM.

VAN DE:
  CLIP:  "anh nay GIONG gi?" (so sanh, khong tra loi)
  YOLO:  "co GI trong anh?" (chi phat hien vat the)
  VQA:   "anh nay dang LAM GI? O DAU? MAU GI?"
         → Tra loi bang cau hoan chinh

  VD: "What is the person wearing?" → "A red shirt and blue jeans"

CACH LAM:
  Dung model multimodal (hieu ca anh + text):
  - GPT-4V (OpenAI) — API tra phi
  - Gemini (Google) — FREE 15 req/min
  - LLaVA (local) — chay offline, can GPU

  Bai nay dung Gemini (FREE, khong can GPU).

CACH CHAY:
  pip install google-generativeai
  Lay API key FREE tai: https://aistudio.google.com/apikey
  Dat bien moi truong: export GEMINI_API_KEY="your-key-here"
  conda activate aic && python learn/vqa1_demo.py

LUU Y: Neu chua co API key, script se chay o che do DEMO
  (chi in huong dan, khong goi API).
"""
import os
import glob
import json
import time

# =================================================================
# BUOC 1: KIEM TRA API KEY
# =================================================================
api_key = os.environ.get("GEMINI_API_KEY", "")

if not api_key:
    print("=" * 60)
    print("VISUAL Q&A — HUONG DAN SETUP")
    print("=" * 60)
    print()
    print("De dung VQA, can Gemini API key (FREE):")
    print()
    print("1. Vao: https://aistudio.google.com/apikey")
    print("2. Dang nhap Google, bam 'Create API Key'")
    print("3. Copy key")
    print("4. Chay:")
    print("   export GEMINI_API_KEY=\"your-key-here\"")
    print("   python learn/vqa1_demo.py")
    print()
    print("Hoac them vao file .env:")
    print("   echo 'GEMINI_API_KEY=your-key' >> .env")
    print()
    print("FREE: 15 request/phut, 1500 request/ngay")
    print("=" * 60)
    print()
    print("Dang chay o che do DEMO (khong goi API)...")
    print()

# =================================================================
# BUOC 2: TIM KEYFRAME
# =================================================================
keyframe_dir = "data/keyframes"
image_paths = sorted(glob.glob(f"{keyframe_dir}/**/*.jpg", recursive=True))

if not image_paths:
    print("KHONG TIM THAY KEYFRAME. Chay keyframe1_extract.py truoc.")
    exit()

print(f"Tim thay {len(image_paths)} keyframe.\n")

# =================================================================
# BUOC 3: VQA FUNCTION
# =================================================================
def ask_about_image(image_path, question, max_retries=3):
    """Hoi 1 cau ve 1 anh, tu dong retry neu bi rate limit."""
    import google.generativeai as genai
    from PIL import Image

    genai.configure(api_key=api_key)
    gmodel = genai.GenerativeModel("gemini-2.0-flash")

    img = Image.open(image_path)

    for attempt in range(max_retries):
        try:
            response = gmodel.generate_content([question, img])
            return response.text
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = 60 * (attempt + 1)
                print(f"    Rate limit! Doi {wait}s roi thu lai...")
                time.sleep(wait)
            else:
                raise

# =================================================================
# BUOC 4: DEMO
# =================================================================
if api_key:
    print("Co API key! Dang demo VQA...\n")

    # Hoi ve 3 keyframe dau tien, moi anh 1 cau thoi (tiet kiem quota)
    question = "Describe this image in detail: what objects, people, activities, colors, and location do you see?"

    vqa_results = {}

    for i, path in enumerate(image_paths[:3]):
        name = os.path.basename(path)
        print(f"--- {name} ---")
        print(f"  Q: {question}")
        try:
            answer = ask_about_image(path, question)
            print(f"  A: {answer[:300]}")
            vqa_results[name] = {"description": answer}
        except Exception as e:
            print(f"  Loi: {e}")
            vqa_results[name] = {"description": f"Error: {e}"}
        print()
        time.sleep(5)  # doi 5s giua moi request, tranh rate limit

    # Luu ket qua
    vqa_path = "data/vqa_results.json"
    with open(vqa_path, "w", encoding="utf-8") as f:
        json.dump(vqa_results, f, ensure_ascii=False, indent=2)
    print(f"\nDa luu vao: {vqa_path}")

else:
    print("--- CHE DO DEMO (khong co API key) ---")
    print()
    print("Neu co API key, script se:")
    print(f"  1. Lay {min(3, len(image_paths))} keyframe dau tien")
    print("  2. Hoi Gemini 3 cau ve moi anh:")
    print("     - Mo ta chi tiet anh")
    print("     - Mau sac chinh")
    print("     - Dia diem chup")
    print("  3. Luu ket qua vao data/vqa_results.json")
    print()
    print("Vi du output:")
    print('  Q: "Describe this image"')
    print('  A: "A busy street market in Vietnam with vendors selling')
    print('       tropical fruits. Several motorbikes are parked nearby..."')

print()
print("=" * 50)
print("CACH DUNG TRONG HE THONG THI:")
print("  User hoi: 'Tim canh co nguoi mac ao do dang nau an'")
print("  → CLIP tim top 20 canh 'person cooking'")
print("  → VQA hoi tung anh: 'Is there a person wearing red cooking?'")
print("  → Loc lai nhung anh tra loi 'yes'")
print("  → Tra ve anh chinh xac nhat")
print()
print("  User hoi: 'Bien so xe trong anh la gi?'")
print("  → OCR co the doc sai → VQA doc chinh xac hon")
print("=" * 50)
