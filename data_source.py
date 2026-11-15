"""Fetch the original UCI file and verify the exact CSV used in the project."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "Occupancy_Estimation.csv"
URL = "https://archive.ics.uci.edu/static/public/864/room+occupancy+estimation.zip"
DIGEST = "c090ee5c94b61762bfec9d22767864490b51030f98cca030f02822468db25d9c"


def load_bytes():
    if DATA.exists():
        payload = DATA.read_bytes()
    else:
        with urlopen(URL, timeout=30) as response:
            with ZipFile(BytesIO(response.read())) as archive:
                payload = archive.read(DATA.name)
    if sha256(payload).hexdigest() != DIGEST:
        raise ValueError("CSV checksum mismatch. Restore the original UCI CSV.")
    return payload


if __name__ == "__main__":
    payload = load_bytes()
    DATA.parent.mkdir(exist_ok=True)
    DATA.write_bytes(payload)
    print(DATA)
