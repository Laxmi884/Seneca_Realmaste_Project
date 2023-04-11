import pandas as pd

from base.util import selectFromTwo


class WriteBackMixin:

    def writeback_both_saletp(self, both=True):
        """ To writeback only original saletp, set both to False"""
        self._writeback_both_saletp = both

    def prepare_predict_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prepare the predicting data.
        1. Fill the missing values with the mean value of the column.
        2. limit columns to the columns in the model.
        """
        self.generate_numeric_columns()
        Xsub = X.loc[:, self.x_numeric_columns_]
        return Xsub.fillna(Xsub.mean(), inplace=False)

    def get_writeback(self, df: pd.DataFrame, orig_col: str = None) -> pd.Series:
        """get writeback result"""
        print(f'get_writeback {df.shape}')
        df = self.prepare_predict_data(df)
        y = self.predict(df)
        if (orig_col is not None) and (orig_col in df.columns):
            print(f'use original col first: {orig_col}')
            y = df.loc[:, orig_col].combine(y, selectFromTwo).astype(int)
        return y

    def writeback(
        self,
        new_col,
        date_span: int = None,
        suffix_list: list[str] = None,
        orig_col: str = None,
    ) -> pd.DataFrame:
        """Write back the results to the source"""
        print(f'writeback {self.name}')
        scale = self.scale
        # default is to writeback both saletp
        if getattr(self, '_writeback_both_saletp', True):
            scale = scale.copy()
            scale.sale = None
        self.data_source.writeback(
            self,
            new_col,
            scale=self.scale,
            cols=self.col_list,
            date_span=date_span or self.source_date_span,
            suffix_list=suffix_list or self.source_suffix_list,
            orig_col=orig_col
        )
