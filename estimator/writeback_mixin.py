from base.timer import Timer
import pandas as pd


class WritebackMixin:
    """Writeback template class.

    This class is used to write back the results of the estimators to the
    data source.
    Implementations of this class shall override the :meth:`writeback` method. 
    """

    def writeback(
        self,
        new_col: str = None,
        df_grouped: pd.DataFrame = None,
        db_col: str = None,
    ) -> int:
        """Write back the results to the data source.

        Parameters
        ----------
        new_col : str, optional
          the name of the new collection to write the results to. If not provided, 
          the implementation shall write back to the original column with -e.

        Returns
        -------
        the number of records written back.
        """
        if new_col is None:
            new_col = self.get_output_column()
        if db_col is None:
            db_col = self.get_writeback_db_column()
        timer = Timer(new_col, self.logger)
        timer.start()
        counter = 0
        model_key = self.__model_key__()
        for scale in self.scales.values():

            if not hasattr(scale.meta, model_key):
                continue
            model_dict = scale.meta[model_key]
            if model_dict['model'] is None:
                continue
            df = self.load_data(scale=scale)
            # fill missing values with mean
            df.fillna(df.mean())
            x_cols = model_dict['x_cols']
            y = model_dict['model'].predict(df[x_cols])
            df[new_col] = y
            self.data_source.writeback(
                col=new_col,
                y=df[new_col],
                df_grouped=df_grouped,
                db_col=db_col,
            )
            counter += len(y)
        timer.stop(counter)
        return counter
