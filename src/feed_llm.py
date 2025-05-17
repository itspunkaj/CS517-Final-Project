import pandas as pd
import subprocess
import os

# Paths - Adjust accordingly
CSV_PATH = "../data/train/retrieved_articles_text.csv"
IMAGES_DIR = "../data/big_data/database_images_compressed90"  # folder containing images named like: {imageID}.jpg
OUTPUT_CSV = "../data/train/enriched_captions.csv"
OLLAMA_MODEL = "llava"  # change if your model has a different name

def build_prompt(image_path, article_texts):
    prompt = (
        "You are an AI assistant trained to analyze images and generate detailed captions. "
        "Below are 10 related article excerpts (ranked from most to least relevant).\n\n"
        "### Articles:\n"
    )
    for i, text in enumerate(article_texts, start=1):
        prompt += f"{i}. {text}\n\n"
    prompt += (
        "### Task:\n"
        f"Generate an enriched caption for the image at '{image_path}' by incorporating "
        "key details from the articles (prioritizing higher-ranked ones).\n"
        "Be descriptive and factual."
    )
    return prompt

def query_ollama(prompt):
    process = subprocess.Popen(
        ['ollama', 'run', OLLAMA_MODEL],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    out, err = process.communicate(prompt)
    if err:
        print("Ollama error:", err)
    return out.strip()

def main():
    df = pd.read_csv(CSV_PATH)
    enriched_captions = []

    for idx in range(len(df) - 1, -1, -1):
        row = df.iloc[idx]
        image_id = row["imageID"]
        article_texts = [row[f"text{i}"] for i in range(1, 11)]

        image_path = os.path.join(IMAGES_DIR, f"{image_id}.jpg")
        if not os.path.exists(image_path):
            print(f"⚠️ Image not found: {image_path}, skipping {image_id}")
            continue

        prompt = build_prompt(image_path, article_texts)
        print(f"\nProcessing image {image_id} ({len(df) - idx}/{len(df)})...")
        # print("Generated prompt:", prompt[:200] + "...")  # Print first 200 chars of prompt

        caption = query_ollama(prompt)
        print("\nGenerated Caption:")
        print("------------------")
        print(caption)
        print("------------------")
        
        enriched_captions.append({"imageID": image_id, "caption": caption})

        out_df = pd.DataFrame(enriched_captions)
        out_df.to_csv(OUTPUT_CSV, index=False)

    # Save to CSV
    out_df = pd.DataFrame(enriched_captions)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Enriched captions saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
