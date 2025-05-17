import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import open_clip
from torchvision import transforms

from transformers import pipeline

# Use 0 for GPU, -1 for CPU as per Hugging Face convention
# summarizer = pipeline(
#     "summarization", 
#     model="facebook/bart-large-cnn", 
#     device=-1,
# )
ummarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=0 if torch.cuda.is_available() else -1)



device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# STEP 1: Load Article Data
# ---------------------------
print("Loading articles...")
article_df = pd.read_csv("../data/big_data/sampled_positive_pairs.csv")

# Clean text
article_df["title"] = article_df["title"].fillna("").apply(lambda x: x.encode("ascii", "ignore").decode())
article_df["content"] = article_df["content"].fillna("").apply(lambda x: x.encode("ascii", "ignore").decode())

def first_50_words(text):
    return " ".join(text.split()[:50])

def summarize_text(text, max_chunk_words=300, max_summary_length=100):
    # Truncate to manageable length if needed
    words = text.split()
    chunks = [ " ".join(words[i:i + max_chunk_words]) for i in range(0, len(words), max_chunk_words) ]
    summaries = []

    for chunk in chunks:
        try:
            summary = summarizer(chunk, max_length=max_summary_length, min_length=25, do_sample=False)[0]['summary_text']
            summaries.append(summary)
        except Exception as e:
            print("Summarization error:", e)
            continue

    return " ".join(summaries)

# Apply summarization
tqdm.pandas(desc="Summarizing content")
article_df["content_short"] = article_df["content"].progress_apply(lambda x: summarize_text(x))

# article_df["content_short"] = article_df["content"].apply(first_50_words)
# article_df["text"] = article_df["title"] + " " + article_df["content_short"]
article_df['text'] = article_df["title"]
article_text_map = article_df.set_index("article_id")["text"].to_dict()

# ---------------------------
# STEP 2: Load Triplets
# ---------------------------
print("Loading triplets...")
triplet_df = pd.read_csv("../data/big_data/sampled_triplets.csv")

# ---------------------------
# STEP 3: Load CLIP Model
# ---------------------------
print("Loading CLIP model...")
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai', device=device
)
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# ---------------------------
# STEP 4: Encode Articles (Text)
# ---------------------------
print("Encoding articles with CLIP...")
article_embeddings = {}
all_article_ids = article_df["article_id"].unique()

with torch.no_grad():
    for aid in tqdm(all_article_ids, desc="Encoding articles"):
        text = article_text_map.get(aid, "")
        if not text.strip():
            article_embeddings[aid] = np.zeros(512)
            continue
        tokens = tokenizer([text]).to(device)
        text_features = model.encode_text(tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        article_embeddings[aid] = text_features.squeeze(0).cpu().numpy()

# ---------------------------
# STEP 5: Encode Images
# ---------------------------
print("Encoding images with CLIP...")
image_embeddings = {}
image_ids = triplet_df["image_id"].unique()

with torch.no_grad():
    for img_id in tqdm(image_ids, desc="Encoding images"):
        img_path = f"../data/big_data/database_images_compressed90/{img_id}.jpg"
        if not os.path.exists(img_path):
            image_embeddings[img_id] = np.zeros(512)
            continue
        try:
            img = Image.open(img_path).convert("RGB")
            img_input = preprocess(img).unsqueeze(0).to(device)
            img_features = model.encode_image(img_input)
            img_features /= img_features.norm(dim=-1, keepdim=True)
            image_embeddings[img_id] = img_features.squeeze(0).cpu().numpy()
        except Exception as e:
            print(f"Error processing image {img_id}: {e}")
            image_embeddings[img_id] = np.zeros(512)

# ---------------------------
# STEP 6: Combine Triplets
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
final_df.to_csv("../data/big_data/triplet_training_data_new.csv", index=False)
print("triplet_training_data_new.csv saved.")
