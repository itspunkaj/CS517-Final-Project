import csv
from pycocoevalcap.cider.cider import Cider
import re
import torch
from transformers import CLIPProcessor, CLIPModel

# File paths
enriched_captions_file = '../data/train/enriched_captions_new.csv'
filtered_gt_file = '../data/train/filtered_gt.csv'
output_file = '../data/train/cider_clip_results_new.csv'

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def preprocess_caption(caption):
    caption = caption.lower()
    caption = re.sub(r'[^\w\s]', '', caption)  # Remove punctuation
    caption = re.sub(r'\s+', ' ', caption).strip()  # Normalize whitespace
    return caption

# Load predicted captions
predicted_captions = {}
with open(enriched_captions_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        image_id = row['imageID']
        predicted_caption = row['caption']
        predicted_captions[image_id] = preprocess_caption(predicted_caption)

# Load ground truth captions
gt_captions = {}
with open(filtered_gt_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        image_id = row['retrieved_image_id']
        gt_caption = row['caption']
        gt_captions[image_id] = preprocess_caption(gt_caption)

# Compute CIDEr and CLIPScore
cider_scorer = Cider()
results = []

from torch.nn.functional import cosine_similarity

def compute_clip_score(gt_text, pred_text):
    # Tokenize both texts
    inputs_gt = clip_processor(text=[gt_text], return_tensors="pt", padding=True,truncation=True).to(device)
    inputs_pred = clip_processor(text=[pred_text], return_tensors="pt", padding=True,truncation=True).to(device)

    # Extract embeddings using only the text model
    with torch.no_grad():
        embedding_gt = clip_model.get_text_features(**inputs_gt)
        embedding_pred = clip_model.get_text_features(**inputs_pred)

    # Normalize embeddings and compute cosine similarity
    embedding_gt = embedding_gt / embedding_gt.norm(dim=-1, keepdim=True)
    embedding_pred = embedding_pred / embedding_pred.norm(dim=-1, keepdim=True)
    similarity = cosine_similarity(embedding_gt, embedding_pred).item()

    return similarity * 100  # Scale for readability


for image_id, predicted_caption in predicted_captions.items():
    if image_id in gt_captions:
        gt_caption = gt_captions[image_id]
        cider_score, _ = cider_scorer.compute_score(
            {0: [gt_caption]},
            {0: [predicted_caption]}
        )
        clip_score = compute_clip_score(gt_caption, predicted_caption)
        print(f"Image ID: {image_id}, CIDEr: {cider_score:.3f}, CLIPScore: {clip_score:.2f}")
        results.append([image_id, predicted_caption, gt_caption, cider_score, clip_score])

# Save results
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image_ID', 'predicted_caption', 'gt_caption', 'CIDEr Value', 'CLIPScore'])
    writer.writerows(results)

print(f"✅ Results saved to {output_file}")
