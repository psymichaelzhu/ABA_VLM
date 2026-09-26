from pathlib import Path
import pandas as pd

# for EmoMadrid
def clean_emomadrid_ratings():
    raw_path = Path("stimuli/EmoMadrid/raw/1_EMindex_2025_07_07(EMindex).csv")
    clean_path = Path("stimuli/EmoMadrid/ratings.csv")

    df = pd.read_csv(
        raw_path,
        sep=";",
        decimal=",",
        skiprows=1,
        encoding="cp1252"
    )

    # Clean column names
    df.columns = (
        df.columns
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # Rename key columns
    df = df.rename(columns={
        "EM CODE": "Image_id",
        "Valence mean": "Valence",
        "Arousal mean": "Arousal"
    })

    # Drop unnecessary columns
    columns_to_drop = [
        "Thumbnail"
    ]
    df = df.drop(columns=columns_to_drop)

    # Replace remaining spaces in column names with underscores
    df.columns = df.columns.str.replace(" ", "_", regex=False)

    df.to_csv(clean_path, index=False)


def main():
    clean_emomadrid_ratings()


if __name__ == "__main__":
    main()