import json
import csv
import random

# Load original large JSON
with open("../data/big_data/database.json", "r") as f:
    data = json.load(f)

# Set number of articles to sample
N = 1000

# Randomly sample N article IDs
sampled_article_ids = random.sample(list(data.keys()), N)

# Create CSV
with open("../data/big_data/sampled_positive_pairs.csv", "w", newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image_id", "article_id", "title", "content"])

    for article_id in sampled_article_ids:
        entry = data[article_id]
        title = entry.get("title", "")
        content = entry.get("content", "")
        image_ids = entry.get("images", [])

        for image_id in image_ids:
            writer.writerow([image_id, article_id, title, content])
