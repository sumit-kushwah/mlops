import argparse
import os
import sys
import pandas as pd
import mlflow
import io

parser = argparse.ArgumentParser(description="Data Pre-Processing")

parser.add_argument("--input", type=str, help="Raw Input file path", required=True)
parser.add_argument(
    "--output", type=str, help="Processed Output file path", required=True
)
parser.add_argument(
    "--mlflow_uri",
    type=str,
    help="MLFlow Tracking URI",
    default="http://localhost:8080",
)

args = parser.parse_args()

print(f"Data Pre-Processing: {args.input} -> {args.output}")

# reading arguments
input_file = args.input
output_path = args.output
mlflow_tracking_uri = args.mlflow_uri

# basic checks
if os.path.exists(input_file):
    print(f"Input file found: {input_file}")
    input_file = os.path.abspath(input_file)
else:
    print(f"Input file not found: {input_file}")
    sys.exit(1)

if os.path.exists(output_path):
    print(f"Output file found: {output_path}")
else:
    print(f"Output path not found: {output_path}")
    print("Creating output path")
    os.makedirs(output_path, exist_ok=True)

output_file = os.path.abspath(output_path)

df = pd.read_csv(input_file)

mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment("Restaurant-Model")

with mlflow.start_run(run_name="Data Preprocessing", nested=True) as run:
    mlflow.set_tag("release.version", "1.0.0")
    mlflow.log_param("Input file path", input_file)
    mlflow.log_param("Rows Before", df.shape[0])
    mlflow.log_param("Columns Before", df.shape[1])
    mlflow.log_param("Duplicate Rows", df.duplicated().sum())

    cols = ["Liquor License Obtained", "Live Sports Rating", "Registration Number"]
    df.drop(cols, axis=1, inplace=True)

    # Replace missing values with median
    cols = [
        "Facebook Popularity Quotient",
        "Instagram Popularity Quotient",
        "Resturant Tier",
        "Live Music Rating",
        "Comedy Gigs Rating",
        "Value Deals Rating",
        "Ambience",
        "Overall Restaurant Rating",
    ]

    for col in cols:
        df[col].fillna(df[col].median(), inplace=True)

    df["Cuisine1"] = df["Cuisine"].apply(lambda x: x.split(",")[0])
    df["Cuisine2"] = df["Cuisine"].apply(lambda x: x.split(",")[1])
    df.drop("Cuisine", axis=1, inplace=True)

    # label encoding for Cuisine1 and Cuisine2
    unique_cuisines = (
        df["Cuisine1"].unique().tolist() + df["Cuisine2"].unique().tolist()
    )

    cuisine_dict = {}
    i = 1
    for cuisine in unique_cuisines:
        if cuisine not in cuisine_dict:
            cuisine_dict[cuisine] = i
            i += 1

    df["Cuisine1"] = df["Cuisine1"].map(cuisine_dict)
    df["Cuisine2"] = df["Cuisine2"].map(cuisine_dict)

    # Label Encoding for other categorical variables
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()

    cols = [
        "City",
        "Restaurant Location",
        "Restaurant Type",
        "Endorsed By",
        "Restaurant Theme",
    ]

    for col in cols:
        df[col] = le.fit_transform(df[col])

    # opening day of restaurant into ordinal date
    df["Opening Day of Restaurant"] = pd.to_datetime(
        df["Opening Day of Restaurant"], format="%d/%m/%y"
    ).apply(lambda x: x.toordinal())

    output = io.StringIO()
    df.info(buf=output)
    info_string = output.getvalue()

    mlflow.log_text(info_string, "cleaned_df_info.text")

    mlflow.log_param("Rows After", df.shape[0])
    mlflow.log_param("Columns After", df.shape[1])
    output_file = os.path.join(output_path, "processed_data.csv")
    df.to_csv(output_file, index=False)
    mlflow.log_artifact(input_file)
    mlflow.log_artifact(output_file)
