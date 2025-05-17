import pandas as pd
import random

# Load sampled positive pairs
df = pd.read_csv("../data/big_data/sampled_positive_pairs.csv")

# Get list of all unique article_ids
all_articles = df["article_id"].unique().tolist()

# Create mapping: image_id -> article_id
image_to_article = dict(zip(df["image_id"], df["article_id"]))

triplets = []

for image_id, pos_article_id in image_to_article.items():
    # Choose a random negative article ≠ positive one
    neg_article_id = random.choice([aid for aid in all_articles if aid != pos_article_id])
    triplets.append((image_id, pos_article_id, neg_article_id))

# Save to CSV
triplet_df = pd.DataFrame(triplets, columns=["image_id", "positive_article_id", "negative_article_id"])
triplet_df.to_csv("../data/big_data/sampled_triplets.csv", index=False)
