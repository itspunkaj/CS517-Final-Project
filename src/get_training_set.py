import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet18
from sentence_transformers import SentenceTransformer

# ---------------------------
# STEP 1: Load Article Data
# ---------------------------
print("Loading articles...")
article_df = pd.read_csv("../data/big_data/sampled_positive_pairs.csv")

# Remove surrogate characters and fill NaNs
article_df["title"] = article_df["title"].fillna("").apply(lambda x: x.encode("ascii", "ignore").decode())
article_df["content"] = article_df["content"].fillna("").apply(lambda x: x.encode("ascii", "ignore").decode())

def first_50_words(text):
    words = text.split()
    return " ".join(words[:50])

article_df["content_short"] = article_df["content"].apply(first_50_words)

article_df["text"] = article_df["title"] + " " + article_df["content_short"]
article_text_map = article_df.set_index("article_id")["text"].to_dict()

# ---------------------------
# STEP 2: Load Triplets
# ---------------------------
print("Loading triplets...")
triplet_df = pd.read_csv("../data/big_data/sampled_triplets.csv")
all_article_ids = pd.unique(triplet_df[["positive_article_id", "negative_article_id"]].values.ravel())

# ---------------------------
# STEP 3: Article Embeddings
# ---------------------------
print("Encoding articles with clip-ViT-B-32...")
text_model = SentenceTransformer("clip-ViT-B-32")
article_embeddings = {}

all_article_ids = article_df["article_id"].unique()
for aid in tqdm(all_article_ids, desc="Encoding articles"):
    text = article_text_map.get(aid, "")
    emb = text_model.encode(text)
    article_embeddings[aid] = emb

# ---------------------------
# STEP 4: Image Embeddings
# ---------------------------
print("Encoding images with clip-ViT-B-32...")

# Use same model as text (CLIP-based)
image_model = SentenceTransformer("clip-ViT-B-32")
image_embeddings = {}
image_ids = triplet_df["image_id"].unique()

for img_id in tqdm(image_ids, desc="Encoding images"):
    img_path = f"../data/big_data/database_images_compressed90/{img_id}.jpg"
    if not os.path.exists(img_path):
        image_embeddings[img_id] = np.zeros(512)
        continue
    try:
        img = Image.open(img_path).convert("RGB")
        emb = image_model.encode(img)
        image_embeddings[img_id] = emb
    except Exception as e:
        print(f"Error processing image {img_id}: {e}")
        image_embeddings[img_id] = np.zeros(512)

# ---------------------------
# STEP 5: Combine Into Final CSV
# ---------------------------
print("Combining triplets into training CSV...")
rows = []

for _, row in tqdm(triplet_df.iterrows(), total=len(triplet_df), desc="Building training set"):
    img_id = row["image_id"]
    pos_id = row["positive_article_id"]
    neg_id = row["negative_article_id"]

    if (
        img_id not in image_embeddings or
        pos_id not in article_embeddings or
        neg_id not in article_embeddings
    ):
        continue

    rows.append({
        "image_id": img_id,
        "image_embedding": ",".join(map(str, image_embeddings[img_id])),
        "positive_article_embedding": ",".join(map(str, article_embeddings[pos_id])),
        "negative_article_embedding": ",".join(map(str, article_embeddings[neg_id])),
        "positive_article_id": pos_id,
        "negative_article_id": neg_id,
    })

final_df = pd.DataFrame(rows)
final_df.to_csv("../data/big_data/triplet_training_data.csv", index=False)
print("triplet_training_data.csv saved.")
