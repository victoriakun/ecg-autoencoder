import wfdb
import os

DATA_DIR = "data/mitbih"
os.makedirs(DATA_DIR, exist_ok=True)

RECORDS = ["100", "101", "102", "103", "104"]


def download_records():
    for rec in RECORDS:
        print(f"Downloading record {rec} ...")
        wfdb.dl_database(
            "mitdb",
            dl_dir=DATA_DIR,
            records=[rec]
        )
    print("Done.")


if __name__ == "__main__":
    download_records()
