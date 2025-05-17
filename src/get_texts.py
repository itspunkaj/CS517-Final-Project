import json
import pandas as pd

# Load article database
with open("../data/big_data/database.json", "r") as f:
    database = json.load(f)

# Load top-10 retrieved articles per image
retrieved_df = pd.read_csv("../data/train/retrieved_articles.csv")

def get_article_text(article_id):
    article = database.get(article_id, None)
    if not article:
        return ""
    title = article.get("title", "")
    content = article.get("content", "")
    # get first 20 words of content
    content_words = content.split()[:20]
    content_snippet = " ".join(content_words)
    # combine title + content snippet
    return f"{title} {content_snippet}"

rows = []
for _, row in retrieved_df.iterrows():
    image_id = row["image_id"]
    texts = []
    for i in range(1, 11):
        article_col = f"article{i}"
        article_id = str(row.get(article_col, ""))
        text = get_article_text(article_id)
        texts.append(text)
    row_dict = {"imageID": image_id}
    for i, text in enumerate(texts, 1):
        row_dict[f"text{i}"] = text
    rows.append(row_dict)

output_df = pd.DataFrame(rows)
output_df.to_csv("../data/train/retrieved_articles_text.csv", index=False)
print("✅ Saved retrieved_articles_text.csv with titles + content snippets.")
