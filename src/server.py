from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import time
import json
import os
import csv

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

uploaded_image_name = None  # store uploaded filename

# Step 3: Read database.json to find articles corresponding to the IDs
database_json_path = '../data/big_data/database.json'  # Update with the correct path
with open(database_json_path, 'r') as json_file:
    database = json.load(json_file)

# Add a GET route
@app.route('/status', methods=['GET'])
def server_status():
    return jsonify({'status': 'Server is running', 'uptime': 'Active'}), 200

@app.route('/upload', methods=['POST'])
def upload_image():
    global uploaded_image_name
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image = request.files['image']
    uploaded_image_name = image.filename  # Save or process as needed
    print("Uploaded image:", uploaded_image_name)

    return jsonify({'status': 'Image uploaded successfully'}), 200

@app.route('/stream', methods=['GET'])
def stream_events():
    def generate_events():
        # Step 1: Send top 10 articles
        articles = [{"id": i, "title": f"Article {i}", "content": f"Content of article {i}"} for i in range(1, 11)]
        yield f"data: {json.dumps({'articles': articles})}\n\n"
        time.sleep(5)

        # Step 2: Send generated caption
        caption = "Generated caption for the image"
        yield f"data: {json.dumps({'caption': caption})}\n\n"
        time.sleep(2)

        # Step 3: Send CIDEr value
        cider_value = 0.85
        yield f"data: {json.dumps({'cider': cider_value})}\n\n"

    return Response(generate_events(), mimetype='text/event-stream')

@app.route('/process_image', methods=['POST'])
def process_image():
    # Step 1: Check if an image is provided
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image = request.files['image']
    image_name = os.path.splitext(image.filename)[0]
    print(f"Processing image: {image_name}")

    # Step 2: Read retrieved_articles.csv to find the image and extract 10 article IDs
    articles_csv_path = '../data/train/retrieved_articles.csv'  # Update with the correct path
    if not os.path.exists(articles_csv_path):
        return jsonify({'error': 'retrieved_articles.csv not found'}), 500

    article_ids = []
    with open(articles_csv_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['image_id'] == image_name:  # Match the image name
                article_ids = [row[f'article{i}'] for i in range(1, 11) if f'article{i}' in row]
                break

    if not article_ids:
        return jsonify({'error': 'No matching articles found for the image'}), 404

    

    articles = []
    for article_id in article_ids:
        if article_id in database:
            articles.append(database[article_id])

    if not articles:
        return jsonify({'error': 'No articles found for the provided IDs'}), 404

    # File paths
    enriched_captions_file = '../data/train/enriched_captions.csv'
    cider_clip_results_file = '../data/train/cider_clip_results.csv'

    # Step 1: Load enriched_captions.csv into a dictionary
    enriched_captions = {}
    with open(enriched_captions_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row['imageID']
            enriched_caption = row['caption']
            enriched_captions[image_id] = enriched_caption

    # Step 2: Find the enriched_caption and CLIPScore for the provided imageID
    result = {}
    with open(cider_clip_results_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row['image_ID']
            clip_score = row['CLIPScore']
            if image_id == image_name:  # Match the provided imageID
                result = {
                    'imageID': image_id,
                    'enriched_caption': enriched_captions.get(image_id, 'No caption found'),
                    'CLIPScore': clip_score
                }
                break  # Exit loop once the matching imageID is found

    # Step 3: Return the result
    results = [result] if result else []

    # Step 4: Return the articles as a JSON array of objects
    return jsonify({'articles': articles,'results':results}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)