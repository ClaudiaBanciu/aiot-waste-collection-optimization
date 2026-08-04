import pandas as pd
import re


class CSVPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_csv(file_path)

    def clean_data(self) -> pd.DataFrame:
        """Clean the data and return the cleaned DataFrame."""
        self._remove_invalid_rows()
        self._create_address_column()
        return self.df

    def _remove_invalid_rows(self):
        missing_street = self.df[self.df["Street"].isna()]
        missing_number = self.df[self.df["Number"].isna()]
        invalid_number = self.df[~self.df["Number"].astype(str).apply(self._is_valid_number)]

        invalid_rows = pd.concat([missing_street, missing_number, invalid_number]).drop_duplicates()
        self.df = self.df.drop(invalid_rows.index)

    def _is_valid_number(self, value):
        if pd.isna(value):
            return False
        value = str(value).strip()
        match = re.match(r'^(\d+)(?:\s?[A-Za-z]?|-.*)?$', value)
        return bool(match)

    def _create_address_column(self):
        self.df["Address"] = (
            self.df["Street"] + " " + self.df["Number"].astype(str) + ", " +
            self.df["City"] + ", " + self.df["Country"]
        )