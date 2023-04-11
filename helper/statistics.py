import pandas as pd

from data.data_source import DataSource

###
# count, mean, std, min, 25%, 50%, 75%, max
###


class BaselineStatistics:
    def __init__(self, data_source: DataSource) -> None:
        self.data_source = data_source
        pass

    def get_stats(self, df, col) -> dict:
        return df[col].describe().to_dict()

    def get_stats_for_columns(self, df, cols) -> pd.DataFrame:
        return df[cols].describe()

    def get_stats_by_group(self, df, group_cols, stats_col) -> pd.DataFrame:
        # https://sparkbyexamples.com/pandas/pandas-get-statistics-for-each-group/#:~:text=groupBy()%20function%20is%20used,max()%20and%20more%20methods.
        return df.groupby(group_cols)[stats_col].describe()

    def calc_stats(
        self,
        df,
        feature_discrete: list[str] = None,
        feature_continuous: list[str] = None,
        col: str = 'lp'
    ) -> dict:
        """Calculate statistics for a sorted dataframe.

        Parameters
        ==========
        df: pd.DataFrame. df_transformed as in data_source.py
            'saletp-b', 'ptype2-l','prov', 'area', 'city','_id' are not indexed.
        """
        stats = {}
        overall = df[col].describe()
        stats['overall'] = overall.to_dict()
        cityDesc = df.groupby(
            level=['prov', 'area', 'city', 'ptype2-l', 'saletp-b'])[col].describe()
        # like this
        df[df['saletp-b'] == 0].groupby(['prov', 'area',
                                        'city', 'ptype2-l', 'br_plus-n'])['lp'].describe()
        df2 = self.data_source.df_transformed
        df2 = df2.mask(df2['sp'] == 0)
        df3 = df2[df2['saletp-b'] == 0]
        dfg = df3.groupby(
            ['prov', 'area', 'city', 'ptype2-l', 'onD-month-n'])
        result = dfg['sp'].describe()
        return stats, cityDesc
