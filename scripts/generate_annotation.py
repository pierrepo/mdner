import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_data():
    """Retrieve our data and loads it into the pd.DataFrame object.
    Returns
    -------
    dict
        returns a tuple containing the pd.DataFrame objects of our datasets.
    """
    datasets = pd.read_parquet(
        "https://sandbox.zenodo.org/record/1169962/files/datasets.parquet"
    )
    gro = pd.read_parquet(
        "https://sandbox.zenodo.org/record/1169962/files/gromacs_gro_files.parquet"
    )
    gro_data = pd.merge(
        gro,
        datasets,
        how="left",
        on=["dataset_id", "dataset_origin"],
        validate="many_to_one",
    )
    return gro_data


def description_length(df, threshold):
    """
    Select datasets from the datasets dataframe that have a description length
    greater than the threshold.

    Parameters
    ----------
    df: pandas.DataFrame
        The datasets dataframe.
    gro: pandas.DataFrame
        The gro dataframe.
    threshold: int
        The threshold for the description length.

    Returns
    -------
    pandas.DataFrame
        The selected datasets dataframe.
    """
    df.loc[:, "description_length"] = (
        df["description"].str.len() + df["title"].str.len()
    )
    if df["description_length"].isnull().values.any():
        df = df.dropna(subset=["description_length", "title", "description"])
    data = df[df["description_length"] > threshold]
    print("[Func. description_length] Number of description : ", data.shape[0])
    return data


def homogeneous_composition(df):
    lipid_data = df[(df["has_protein"] == False) & (df["has_lipid"] == True)]
    protein_data = df[(df["has_protein"] == True) & (df["has_lipid"] == False)]
    size_sample = min(lipid_data.shape[0], protein_data.shape[0])
    sample_lipid = lipid_data.sample(n=size_sample)
    sample_protein = protein_data.sample(n=size_sample)
    data = pd.concat([sample_lipid, sample_protein], ignore_index=True)
    print(
        "[Func. homogeneous_composition] Number of description : ",
        data.shape[0],
    )
    return data


def corpus_similarity(df, threshold):
    cols = ["title", "description"]
    df["corpus"] = df[cols].apply(lambda row: " ".join(row.values.astype(str)), axis=1)
    vectorizer = TfidfVectorizer()
    trsfm = vectorizer.fit_transform(df["corpus"])
    cos_data = pd.DataFrame(
        [
            cosine_similarity(trsfm[i : i + 1], trsfm)[0]
            for i in range(len(df["corpus"]))
        ]
    )
    similarity_data = [(sum(cos_data.iloc[i, :]), i) for i in range(len(cos_data))]
    similarity_data.sort()
    if threshold < len(similarity_data):
        data = df.iloc[[sim[1] for sim in similarity_data[: threshold + 1]], :]
    else:
        data = df
    print(
        "[Func. corpus_similarity] Number of description : ",
        data.shape[0],
    )
    return data


def clear_annotation(df):
    df = df.replace("_", " ", regex=True)
    df["annotation"] = df["annotation"].str.replace(
        r'https?://[^\s<>"]+|www\.[^\s<>"]+',
        "https://removed",
        regex=True,
    )
    return df


def generate_annotation(df):
    path = "../annotations/test/"
    for i in range(len(df)):
        with open(
            path + df.loc[i, "dataset_origin"] + "_" + df.loc[i, "dataset_id"] + ".txt",
            "w",
        ) as f:
            f.write(df.loc[i, "annotation"])


if __name__ == "__main__":
    gro_data = load_data()
    # Remove the duplicate description
    gro_data = gro_data.drop_duplicates("description")
    threshold_length = 600
    threshold_similarity = 200
    # Prerequise for the annotation
    df_length = description_length(gro_data, threshold_length)
    df_similarity = corpus_similarity(df_length, threshold_similarity)
    df_composition = homogeneous_composition(df_similarity)
    # Setup and cleaning up the annotation
    df = df_composition[["dataset_id", "dataset_origin"]]
    df = df.copy()
    df["annotation"] = df_composition["title"] + df_composition["description"]
    data = clear_annotation(df)
    generate_annotation(data)
