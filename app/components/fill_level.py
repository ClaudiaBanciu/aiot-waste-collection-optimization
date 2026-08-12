import pandas as pd


class FillLevel:
    """Normalizes the fill level column of a waste container DataFrame.

    Handles both text ('68%') and decimal (0.68) representations,
    converting everything to integers in the 0–100 range and storing
    the result in a new 'Fill_num' column.

    Usage:
        fl = FillLevel(df)
        df = fl.data          # DataFrame with the added Fill_num column
    """

    SOURCE_COLUMN = "fill_level"
    TARGET_COLUMN = "Fill_num"

    def __init__(self, df: pd.DataFrame):
        self._df = df.copy()
        self._df[self.TARGET_COLUMN] = (
            self._normalize(self._df[self.SOURCE_COLUMN]).astype("Int64")
        )

    @property
    def data(self) -> pd.DataFrame:
        """DataFrame with the normalized Fill_num column added."""
        return self._df

    @staticmethod
    def _normalize(series: pd.Series) -> pd.Series:
        """Convert a fill-level series to numeric values in the 0–100 range."""
        if series.dtype == object:
            return pd.to_numeric(series.astype(str).str.rstrip("%"), errors="coerce")
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.max() <= 1:
            numeric = numeric * 100
        return numeric
