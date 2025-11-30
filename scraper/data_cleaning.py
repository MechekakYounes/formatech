import pandas as pd 
import os 
import html

def clean_data(file_path: str) -> pd.DataFrame:
    """
    Cleans the raw StackOverflow questions data by removing duplicates and handling missing values
    removing html tags from titles
    normalizing characteres into lowercase

    """
    df = pd.read_csv(file_path)

    # rremove duplicates based on question_id
    df = df.drop_duplicates(subset=['question_id'])

    df['title'] = df['title'].apply(html.unescape)  # remove html tags and entities
    df['title'] = df['title'].str.replace(r'$eq', '=', regex=True) # replace $eq with = 
    df['title'] = df['title'].str.lower()

    return df

if __name__ == "__main__":
    raw_dir_path = 'C:\\Users\\Administrator\\Desktop\\Bi\\facebook_sentiment_intelligence\\scraper\\raw_datasets\\'
    for file in os.listdir(raw_dir_path):
        if file.endswith("_questions.csv"):
             raw_file_path = os.path.join(raw_dir_path, file)
             cleaned_df = clean_data(raw_file_path)
             cleaned_file_path = raw_file_path.replace("raw_datasets", "cleaned_datasets").replace(".csv", "_cleaned.csv")
             cleaned_df.to_csv(cleaned_file_path, index=False)
             print(f"Cleaned data saved to {cleaned_file_path}")

   