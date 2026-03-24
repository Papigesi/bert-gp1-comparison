import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split


def load_imdb_split(base_path, split_name):
    data = []

    for label_name, label_id in [("neg", 0), ("pos", 1)]:
        folder_path = os.path.join(base_path, split_name, label_name)
        file_paths = glob.glob(os.path.join(folder_path, "*.txt"))

        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().strip()

            data.append({
                "text": text,
                "label": label_id
            })
        
    df = pd.DataFrame(data)

    return df

    
def load_imdb_data(base_path="aclImdb", val_size=0.2, random_state=42):
    train_df = load_imdb_split(base_path, "train")
    test_df = load_imdb_split(base_path, "test")
    
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_size,
        random_state=random_state,
        stratify=train_df["label"]
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, val_df, test_df


if __name__ == "__main__":
    train_df, val_df, test_df = load_imdb_data()

    print("=" * 30)
    print("Train Size: ", len(train_df))
    print("Validation Size: ", len(val_df))
    print("Test Size: ", len(test_df))
    print("=" * 30)

    print("\nTrain Sample:")
    print(train_df.head())