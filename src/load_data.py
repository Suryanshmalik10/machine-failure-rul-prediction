
from pathlib import Path
import pandas as pd

COLUMNS = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

def read_cmapss(name, data_dir):
    path = Path(data_dir) / name

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            "Download the NASA C-MAPSS FD004 files "
            "and place them in data/raw/."
        )

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=COLUMNS
    )

    if df.shape[1] != 26:
        raise ValueError(
            f"{name}: expected 26 columns, got {df.shape[1]}"
        )

    return df

def load_data(data_dir=None):
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent / "data" / "raw"

    train = read_cmapss("train_FD004.txt", data_dir)
    test = read_cmapss("test_FD004.txt", data_dir)

    rul_path = Path(data_dir) / "RUL_FD004.txt"

    if not rul_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {rul_path}"
        )

    rul = pd.read_csv(
        rul_path,
        sep=r"\s+",
        header=None,
        names=["RUL"]
    )

    # RUL row i corresponds to test engine i+1.
    rul.insert(0, "engine_id", range(1, len(rul) + 1))

    return train, test, rul

def validate_data(train, test, rul):
    expected_shapes = {
        "train": (61249, 26),
        "test": (41214, 26),
        "rul": (248, 2)
    }

    for name, df in [
        ("train", train),
        ("test", test),
        ("rul", rul)
    ]:
        expected = expected_shapes[name]
        if df.shape != expected:
            raise ValueError(
                f"{name}: expected shape {expected}, got {df.shape}"
            )

        if df.isnull().any().any():
            raise ValueError(f"{name}: missing values detected")

    train_engines = sorted(train["engine_id"].unique())
    test_engines = sorted(test["engine_id"].unique())
    rul_engines = rul["engine_id"].tolist()

    if len(train_engines) != 249:
        raise ValueError(
            f"Expected 249 training engines, got {len(train_engines)}"
        )

    if test_engines != list(range(1, 249)):
        raise ValueError(
            "Test engine IDs must be exactly 1 through 248."
        )

    if rul_engines != test_engines:
        raise ValueError(
            "RUL engine IDs do not match test engine IDs."
        )

    if (train["cycle"] < 1).any() or (test["cycle"] < 1).any():
        raise ValueError("Cycle numbers must be positive.")

    for name, df in [("train", train), ("test", test)]:
        for engine_id, group in df.groupby("engine_id"):
            cycles = group["cycle"].tolist()
            expected = list(range(1, len(cycles) + 1))

            if cycles != expected:
                raise ValueError(
                    f"{name}: engine {engine_id} has missing, "
                    "duplicated, or unordered cycles."
                )

    if (rul["RUL"] < 0).any():
        raise ValueError("RUL values must be non-negative.")

    print("All validation checks passed!")

if __name__ == "__main__":
    train, test, rul = load_data()
    validate_data(train, test, rul)

    print("\nTraining shape:", train.shape)
    print("Training engines:", train["engine_id"].nunique())
    print("Test shape:", test.shape)
    print("Test engines:", test["engine_id"].nunique())
    print("RUL shape:", rul.shape)

    print("\nSensor standard deviations:")
    print(train.filter(regex=r"^sensor_").std().sort_values().head())

    print("\nTraining preview:")
    print(train.head())

