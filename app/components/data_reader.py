import pandas as pd


class DataReader:
    """Reads a CSV (or Excel) file regardless of encoding or separator.

    The instance holds the file path and caches the result after the
    first read, so repeated calls to read() are free.

    Usage:
        reader = DataReader("path/to/file.csv")
        df = reader.read()
        # or access the cached result later:
        df = reader.data
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._df: pd.DataFrame | None = None

    def read(self) -> pd.DataFrame:
        """Read and return the file. Result is cached on the instance."""
        if self._df is None:
            self._df = self._parse(self.file_path)
        return self._df

    @property
    def data(self) -> pd.DataFrame | None:
        """The cached DataFrame, or None if read() has not been called yet."""
        return self._df

    @staticmethod
    def _parse(path: str) -> pd.DataFrame:
        """Try multiple encodings and separators until the file is read successfully."""
        with open(path, "rb") as f:
            signature = f.read(4)
        if signature[:2] == b"PK":
            return pd.read_excel(path)
        for enc in ["utf-8-sig", "utf-8", "cp1252", "latin1"]:
            for sep in [",", ";"]:
                try:
                    df = pd.read_csv(path, encoding=enc, sep=sep)
                    if "Address" in df.columns:
                        return df
                except (UnicodeDecodeError, pd.errors.ParserError):
                    continue
        raise SystemExit(f"Could not read {path} with any known encoding/separator.")
