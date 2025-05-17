import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import ast
from tqdm import tqdm

# -----------------------
# Model Definition
# -----------------------
class ScoringModel(nn.Module):
    def __init__(self, emb_dim=512):
        super(ScoringModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(emb_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, img_emb, art_emb):
        x = torch.cat([img_emb, art_emb], dim=1)
        score = self.network(x)
        return score.squeeze(1)

# -----------------------
# Load Data
# -----------------------
print("📄 Loading dataset...")
df = pd.read_csv("../data/big_data/triplet_training_data.csv")

def parse_emb(x):
    return np.array(ast.literal_eval(x), dtype=np.float32)

df["image_embedding"] = df["image_embedding"].apply(parse_emb)
df["positive_article_embedding"] = df["positive_article_embedding"].apply(parse_emb)
df["negative_article_embedding"] = df["negative_article_embedding"].apply(parse_emb)

# -----------------------
# Build Article Embedding Pool
# -----------------------
article_id_to_emb = {}
for _, row in df.iterrows():
    article_id_to_emb[row["positive_article_id"]] = row["positive_article_embedding"]
    article_id_to_emb[row["negative_article_id"]] = row["negative_article_embedding"]

candidate_article_ids = list(article_id_to_emb.keys())
candidate_article_embs = torch.tensor([article_id_to_emb[aid] for aid in candidate_article_ids])

# -----------------------
# Load Model
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ScoringModel(emb_dim=512)
model.load_state_dict(torch.load("../model/model.pth", map_location=device))
model.to(device)
model.eval()

# -----------------------
# Helper Function: Top-K Retrieval
# -----------------------
def get_top_k_articles(model, image_emb, candidate_embs, candidate_ids, k=10):
    with torch.no_grad():
        img_tensor = torch.tensor(image_emb).to(device).unsqueeze(0).repeat(len(candidate_embs), 1)
        scores = model(img_tensor, candidate_embs.to(device))
        topk_indices = torch.topk(scores, k=k).indices.cpu().numpy()
        return [(candidate_ids[i], scores[i].item()) for i in topk_indices]

# -----------------------
# Evaluate Top-K Accuracy
# -----------------------
top_k = 10
correct = 0
total = 0

print("🔍 Running Top-K retrieval test...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    img_emb = row["image_embedding"]
    pos_id = row["positive_article_id"]

    top_k_preds = [aid for aid, _ in get_top_k_articles(model, img_emb, candidate_article_embs, candidate_article_ids, k=top_k)]

    if pos_id in top_k_preds:
        correct += 1
    total += 1

accuracy = correct / total
print(f"\n🎯 Top-{top_k} Accuracy: {accuracy:.4f} ({correct}/{total})")
