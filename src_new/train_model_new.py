import pandas as pd
import numpy as np
import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# ----------------------------
# Dataset
# ----------------------------
class TripletDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_emb = np.array(ast.literal_eval(row['image_embedding']), dtype=np.float32)
        pos_emb = np.array(ast.literal_eval(row['positive_article_embedding']), dtype=np.float32)
        neg_emb = np.array(ast.literal_eval(row['negative_article_embedding']), dtype=np.float32)
        return (
            torch.tensor(image_emb),
            torch.tensor(pos_emb),
            torch.tensor(neg_emb)
        )

# ----------------------------
# Scoring Model (Embedder)
# ----------------------------
class ScoringModel(nn.Module):
    def __init__(self, emb_dim=512):
        super(ScoringModel, self).__init__()
        self.img_encoder = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.art_encoder = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def embed_image(self, img_emb):
        return F.normalize(self.img_encoder(img_emb), dim=1)

    def embed_article(self, art_emb):
        return F.normalize(self.art_encoder(art_emb), dim=1)

# ----------------------------
# Training Loop
# ----------------------------
def train_triplet_model(csv_path, epochs=10, batch_size=64, lr=1e-3, emb_dim=512, device='cpu', save_path="../model/model_triplet_embed.pth"):
    dataset = TripletDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = ScoringModel(emb_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.TripletMarginLoss(margin=0.3)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for img_emb, pos_emb, neg_emb in dataloader:
            img_emb = img_emb.to(device)
            pos_emb = pos_emb.to(device)
            neg_emb = neg_emb.to(device)

            # Get learned embeddings
            img_vec = model.embed_image(img_emb)
            pos_vec = model.embed_article(pos_emb)
            neg_vec = model.embed_article(neg_emb)

            loss = criterion(img_vec, pos_vec, neg_vec)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

        # Recall@K after each epoch
        recall = recall_at_k(model, dataset.df, candidate_article_ids, candidate_article_embs, k=10, device=device)
        print(f"Recall@10: {recall:.4f}")

    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    return model

# ----------------------------
# Recall@K
# ----------------------------
def recall_at_k(model, df, candidate_ids, candidate_embs, k=10, device='cpu'):
    correct = 0
    total = 0
    for _, row in df.iterrows():
        img_emb = np.array(ast.literal_eval(row['image_embedding']), dtype=np.float32)
        pos_id = row['positive_article_id']
        top_k = get_top_k_articles(model, img_emb, candidate_embs, candidate_ids, k=k, device=device)
        retrieved_ids = [x[0] for x in top_k]
        if pos_id in retrieved_ids:
            correct += 1
        total += 1
    return correct / total

# ----------------------------
# Article Ranking
# ----------------------------
def get_top_k_articles(model, image_emb, candidate_article_embs, candidate_article_ids, k=10, device='cpu'):
    model.eval()
    with torch.no_grad():
        image_tensor = torch.tensor(image_emb, dtype=torch.float32, device=device).unsqueeze(0)
        image_vec = model.embed_image(image_tensor).repeat(len(candidate_article_embs), 1)
        article_tensor = torch.tensor(candidate_article_embs, dtype=torch.float32, device=device)
        article_vec = model.embed_article(article_tensor)
        scores = F.cosine_similarity(image_vec, article_vec, dim=1).cpu().numpy()
    ranked_indices = np.argsort(-scores)
    return [(candidate_article_ids[i], scores[i]) for i in ranked_indices[:k]]

# ----------------------------
# Candidate Article Pool Builder
# ----------------------------
def build_candidate_articles(csv_path):
    df = pd.read_csv(csv_path)
    pos_articles = df[['positive_article_id', 'positive_article_embedding']].drop_duplicates()
    pos_articles.columns = ['article_id', 'embedding']
    neg_articles = df[['negative_article_id', 'negative_article_embedding']].drop_duplicates()
    neg_articles.columns = ['article_id', 'embedding']
    all_articles = pd.concat([pos_articles, neg_articles]).drop_duplicates('article_id').reset_index(drop=True)
    all_articles['embedding_np'] = all_articles['embedding'].apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float32))
    return all_articles['article_id'].tolist(), np.stack(all_articles['embedding_np'].values)

# ----------------------------
# Usage Example
# ----------------------------
K = 10
csv_path = "../data/big_data/triplet_training_data_new.csv"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Build candidate article pool *before* training (used inside training)
candidate_article_ids, candidate_article_embs = build_candidate_articles(csv_path)

# Train the model
model = train_triplet_model(csv_path, epochs=30, batch_size=64, device=device)

# Get top-K articles for one example
df = pd.read_csv(csv_path)
example_image_emb = np.array(ast.literal_eval(df.iloc[1]['image_embedding']), dtype=np.float32)
example_image_id = df.iloc[1]['image_id']
top_k = get_top_k_articles(model, example_image_emb, candidate_article_embs, candidate_article_ids, k=K, device=device)

print(f"\nTop {K} articles for example image ID: {example_image_id}")
for aid, score in top_k:
    print(f"Article ID: {aid}, Score: {score:.4f}")
