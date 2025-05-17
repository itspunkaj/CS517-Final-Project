import pandas as pd
import subprocess
import os

# Paths - Adjust accordingly
CSV_PATH = "../data/train/retrieved_articles_text_new.csv"
IMAGES_DIR = "../data/big_data/database_images_compressed90"  # folder containing images named like: {imageID}.jpg
OUTPUT_CSV = "../data/train/enriched_captions_new.csv"
OLLAMA_MODEL = "llava"  # change if your model has a different name

def build_prompt(image_path, article_texts):
    prompt = (
        "You are an advanced AI assistant trained to analyze images and synthesize detailed, enriched captions. "
        "Your task is to generate a descriptive and factual caption for the given image by incorporating key details "
        "from the provided article excerpts. Prioritize higher-ranked articles while ensuring the caption is coherent and relevant.\n\n"
        "### Context:\n"
        "Below are 3 related article excerpts ranked from most to least relevant. Use these excerpts to enrich the caption:\n\n"
    )

    for i, text in enumerate(article_texts, start=1):
        prompt += f"{i}. {text}\n\n"

    prompt += (
        "### Task:\n"
        f"Analyze the image located at '{image_path}' and generate a detailed caption that:\n"
        "- Incorporates key details from the articles, prioritizing higher-ranked ones.\n"
        "- Is descriptive, factual, and contextually relevant to the image.\n"
        "- Avoids repetition and ensures clarity.\n\n"
        "### Deliverable:\n"
        "Provide a single enriched caption for the image."
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
        article_texts = [row[f"text{i}"] for i in range(1, 3)]

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
