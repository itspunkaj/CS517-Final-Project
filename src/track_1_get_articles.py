import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import ast
from tqdm import tqdm

# Load filtered GT
filtered_gt_df = pd.read_csv("../data/train/filtered_gt.csv")
retrieved_image_ids = filtered_gt_df["retrieved_image_id"].unique()

# Load triplet training data
triplet_df = pd.read_csv("../data/big_data/triplet_training_data.csv")

def parse_emb(x):
    return np.array(ast.literal_eval(x), dtype=np.float32)

triplet_df["image_embedding"] = triplet_df["image_embedding"].apply(parse_emb)
triplet_df["positive_article_embedding"] = triplet_df["positive_article_embedding"].apply(parse_emb)
triplet_df["negative_article_embedding"] = triplet_df["negative_article_embedding"].apply(parse_emb)

# Map image IDs to embeddings
image_emb_map = {}
for _, row in triplet_df.iterrows():
    image_emb_map[row["image_id"]] = row["image_embedding"]

# Build article embedding pool
article_emb_map = {}
for _, row in triplet_df.iterrows():
    article_emb_map[row["positive_article_id"]] = row["positive_article_embedding"]
    article_emb_map[row["negative_article_id"]] = row["negative_article_embedding"]

candidate_article_ids = list(article_emb_map.keys())
candidate_article_embs = torch.tensor([article_emb_map[aid] for aid in candidate_article_ids], dtype=torch.float32)

# Define model
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ScoringModel(emb_dim=512)
model.load_state_dict(torch.load("../model/model.pth", map_location=device))
model.to(device)
model.eval()

def get_top_k_articles(model, image_emb, candidate_embs, candidate_ids, k=10):
    with torch.no_grad():
        img_tensor = torch.tensor(image_emb, dtype=torch.float32).to(device).unsqueeze(0)
        candidate_embs = candidate_embs.to(device)
        img_tensor = img_tensor.repeat(len(candidate_embs), 1)
        scores = model(img_tensor, candidate_embs)
        topk_indices = torch.topk(scores, k=k).indices.cpu().numpy()
        return [candidate_ids[i] for i in topk_indices]

print("Getting top-10 articles for retrieved images...")

rows = []
for img_id in tqdm(retrieved_image_ids):
    if img_id not in image_emb_map:
        print(f"Image ID {img_id} not found in training embeddings. Skipping.")
        continue
    img_emb = image_emb_map[img_id]
    top_10 = get_top_k_articles(model, img_emb, candidate_article_embs, candidate_article_ids, k=10)
    row = {"image_id": img_id}
    for i, aid in enumerate(top_10, start=1):
        row[f"article{i}"] = aid
    rows.append(row)

# Save to CSV
retrieved_df = pd.DataFrame(rows)
retrieved_df.to_csv("../data/train/retrieved_articles.csv", index=False)
print(f"✅ Saved retrieved articles to ../data/train/retrieved_articles.csv")
