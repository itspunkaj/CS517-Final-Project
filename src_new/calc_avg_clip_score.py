import csv

# File path to cider_clip_results.csv
cider_clip_results_file = '../data/train/cider_clip_results_new.csv'

def calculate_avg_clip_score(file_path):
    total_clip_score = 0
    count = 0

    # Open the CSV file and read the CLIPScore column
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                clip_score = float(row['CLIPScore'])  # Convert CLIPScore to float
                total_clip_score += clip_score
                count += 1
            except ValueError:
                print(f"Invalid CLIPScore value in row: {row}")

    # Calculate the average CLIPScore
    if count == 0:
        return 0  # Avoid division by zero
    return total_clip_score / count

# Calculate and print the average CLIPScore
average_clip_score = calculate_avg_clip_score(cider_clip_results_file)
print(f"Average CLIPScore: {average_clip_score:.2f}")