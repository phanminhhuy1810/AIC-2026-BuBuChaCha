"""Kiem tra moi truong AI Challenge da san sang chua. Chay: python verify_setup.py"""
import sys


def check(name, fn):
    try:
        result = fn()
        print(f"[OK]   {name}: {result}")
        return True
    except Exception as e:
        print(f"[FAIL] {name}: {e}")
        return False


def main():
    print("=== KIEM TRA MOI TRUONG AI CHALLENGE 2026 ===\n")

    okPython = check("Python version", lambda: sys.version.split()[0])

    def checkTorch():
        import torch
        gpu = "GPU: " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only (binh thuong neu may khong co GPU NVIDIA)"
        return f"torch {torch.__version__} | {gpu}"

    def checkClip():
        import torch
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        return "CLIP ViT-B-32 tai duoc"

    def checkFaiss():
        import faiss
        import numpy as np
        index = faiss.IndexFlatIP(4)              # inner product index, 4 dims
        index.add(np.random.rand(10, 4).astype("float32"))
        return f"faiss OK ({index.ntotal} vectors)"

    okTorch = check("PyTorch", checkTorch)
    okTr = check("transformers", lambda: __import__("transformers").__version__)
    okClip = check("open_clip + tai CLIP", checkClip)
    okFaiss = check("FAISS (vector search)", checkFaiss)

    print("\n=== KET QUA ===")
    if all([okPython, okTorch, okTr, okClip, okFaiss]):
        print("Tat ca SAN SANG. Bat dau GD1!")
    else:
        print("Co loi o tren -> cai lai lib bi FAIL: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
