import pandas as pd
import numpy as np
import ast
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


# Dataset class same as before
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

def train_triplet_model(csv_path, epochs=10, batch_size=64, lr=1e-3, emb_dim=512, device='cpu', save_path="../model/model.pth"):
    dataset = TripletDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = ScoringModel(emb_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MarginRankingLoss(margin=1.0)
    # criterion = nn.MSELoss()
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for img_emb, pos_emb, neg_emb in dataloader:
            # print("img_emb shape:", img_emb.shape)
            # print("pos_emb shape:", pos_emb.shape)
            img_emb = img_emb.to(device)
            pos_emb = pos_emb.to(device)
            neg_emb = neg_emb.to(device)
            score_pos = model(img_emb, pos_emb)
            score_neg = model(img_emb, neg_emb)
            target = torch.ones_like(score_pos, device=device)
            loss = criterion(score_pos, score_neg, target)
            # loss = criterion(score_pos, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.10f}")

    if save_path is not None:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    return model


# -- Build candidate article pool from triplets CSV --
def build_candidate_articles(csv_path):
    df = pd.read_csv(csv_path)

    # Get unique positive articles
    pos_articles = df[['positive_article_id', 'positive_article_embedding']].drop_duplicates()
    pos_articles.columns = ['article_id', 'embedding']

    # Get unique negative articles
    neg_articles = df[['negative_article_id', 'negative_article_embedding']].drop_duplicates()
    neg_articles.columns = ['article_id', 'embedding']

    # Combine and drop duplicates again
    all_articles = pd.concat([pos_articles, neg_articles]).drop_duplicates('article_id').reset_index(drop=True)

    # Convert embeddings to np arrays
    all_articles['embedding_np'] = all_articles['embedding'].apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float32))

    article_ids = all_articles['article_id'].tolist()
    article_embs = np.stack(all_articles['embedding_np'].values)

    return article_ids, article_embs

def get_top_k_articles(model, image_emb, candidate_article_embs, candidate_article_ids, k=10, device='cpu'):
    model.eval()
    with torch.no_grad():
        image_tensor = torch.tensor(image_emb, dtype=torch.float32, device=device).unsqueeze(0).repeat(len(candidate_article_embs), 1)
        article_tensor = torch.tensor(candidate_article_embs, dtype=torch.float32, device=device)
        scores = model(image_tensor, article_tensor).cpu().numpy()
    ranked_indices = np.argsort(-scores)
    top_k = [(candidate_article_ids[i], scores[i]) for i in ranked_indices[:k]]
    return top_k


# --- USAGE ---
K=10

csv_path = "../data/big_data/triplet_training_data_new.csv"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Train model
model = train_triplet_model(csv_path, epochs=50, batch_size=64, device=device)

# Build candidate articles pool
candidate_article_ids, candidate_article_embs = build_candidate_articles(csv_path)

# For a given image embedding from your dataset (example first image embedding)
df = pd.read_csv(csv_path)
example_image_emb = np.array(ast.literal_eval(df.iloc[1]['image_embedding']), dtype=np.float32)
example_image_id = df.iloc[1]['image_id']

# Get top-k articles for this image
top_k = get_top_k_articles(model, example_image_emb, candidate_article_embs, candidate_article_ids, k=K, device=device)

print(f"Top {K} articles for example image ID: {example_image_id}")
for aid, score in top_k:
    print(f"Article ID: {aid}, Score: {score:.4f}")
