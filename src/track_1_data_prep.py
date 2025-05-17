import pandas as pd

# Load the ground-truth file
gt_df = pd.read_csv("../data/train/gt_train.csv")

# Load the triplet training data to get the valid image IDs
triplet_df = pd.read_csv("../data/big_data/triplet_training_data.csv")
valid_image_ids = set(triplet_df["image_id"].unique())

# Filter gt.csv to only include rows with retrieved_image_id in the training data
filtered_gt_df = gt_df[gt_df["retrieved_image_id"].isin(valid_image_ids)]

# Save the filtered results
filtered_gt_df.to_csv("../data/train/filtered_gt.csv", index=False)
print(f"✅ Filtered GT saved. Retained {len(filtered_gt_df)} out of {len(gt_df)} rows.")
