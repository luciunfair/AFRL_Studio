"""Build a small balanced sample fragment CSV from public benchmark validation splits.

Example:
    python sample_dataset/sample.py \
        --inputs dataset/ITC_IT_Dataset/unenc_val.csv dataset/EnCoD_Dataset/unenc_val.csv \
                 dataset/ITC_IT_Dataset/enc_val.csv dataset/EnCoD_Dataset/enc_val.csv \
        --samples-per-file 5 \
        --output sample_dataset/sample_dataset.csv \
        --seed 42
"""

import argparse
import os
import pandas as pd


DEFAULT_INPUTS = [
    os.path.join("dataset", "ITC_IT_Dataset", "unenc_val.csv"),
    os.path.join("dataset", "EnCoD_Dataset", "unenc_val.csv"),
    os.path.join("dataset", "ITC_IT_Dataset", "enc_val.csv"),
    os.path.join("dataset", "EnCoD_Dataset", "enc_val.csv"),
]


def build_sample_dataset(input_files, samples_per_file: int, output_file: str, seed: int = 42) -> int:
    all_samples = []
    for idx, file_path in enumerate(input_files):
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Input CSV not found: {file_path}")
        df = pd.read_csv(file_path, header=None)
        if len(df) < samples_per_file:
            raise ValueError(f"{file_path} has only {len(df)} rows (< {samples_per_file}).")
        all_samples.append(df.sample(n=samples_per_file, random_state=seed + idx))

    result = pd.concat(all_samples, ignore_index=True)
    out_dir = os.path.dirname(os.path.abspath(output_file))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    result.to_csv(output_file, index=False, header=False)
    return len(result)


def main():
    parser = argparse.ArgumentParser(description="Create a compact balanced fragment CSV from validation splits.")
    parser.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS, help="Ordered list of headerless fragment CSVs.")
    parser.add_argument("--samples-per-file", type=int, default=5, help="Rows to sample from each input CSV.")
    parser.add_argument("--output", default=os.path.join("sample_dataset", "sample_dataset.csv"), help="Output CSV path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling.")
    args = parser.parse_args()

    total = build_sample_dataset(args.inputs, args.samples_per_file, args.output, seed=args.seed)
    print(f"Done! {total} samples saved to {args.output}")


if __name__ == "__main__":
    main()
