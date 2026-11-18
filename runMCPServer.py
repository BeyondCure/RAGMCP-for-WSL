from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3", device="cpu")
vecs = model.encode(
    ["纯棉面料透气性怎么样？", "真丝面料建议低温熨烫。"],
    normalize_embeddings=True,
    batch_size=8,
)
print(vecs.shape)   # 期望 (2, 1024)