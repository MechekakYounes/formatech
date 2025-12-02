import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd

def plot_topic_distribution(topic_info_csv: str, output_image: str):
    """
    Plots the distribution of topics from the topic modeling summary CSV
    saves the plot as an image file
    """
    df = pd.read_csv(topic_info_csv)
    df = df[df['Topic'] != -1]  # exclude outlier topic
    print ("number of topics:", len(df))

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Topic', y='Count', data=df, legend=False, hue='Topic')
    plt.title('Topic Distribution')
    plt.xlabel('Topic')
    plt.ylabel('Number of Questions')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    print(f"Topic distribution plot saved to {output_image}")

if __name__ == "__main__":
    topic_info_csv = "topic_summary.csv"
    output_image = "topic_distribution.png"
    plot_topic_distribution(topic_info_csv, output_image)