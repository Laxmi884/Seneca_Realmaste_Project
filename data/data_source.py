

from math import isnan
import time
from base.base_cfg import BaseCfg
from numpy import NaN
from data.estimate_scale import EstimateScale
from base.mongo import MongoDB
from base.util import debug, get_utc_datetime_from_str, getUniqueLabels, isNanOrNone, print_dateframe
from datetime import datetime, timedelta
import pandas as pd
from transformer.preprocessor import Preprocessor
from typing import Union
import os
from base.const import CITY_COUNT_THRESHOLD

logger = BaseCfg.getLogger(__name__)

PROV_CITY_TO_AREA = {}
PROV_CITY_TO_AREA_COUNT = {}
PROV_CITY_TO_AREA_DF = None
PROV_CITY_TO_AREA_FILE = 'data/prov_city_to_area.csv'
gEmptyAreaCount = 0


def readProvCityToArea():
    """Read province-city to area mapping from csv file"""
    global PROV_CITY_TO_AREA, PROV_CITY_TO_AREA_COUNT
    # check file exists
    if not (os.path.exists(PROV_CITY_TO_AREA_FILE) and os.path.isfile(PROV_CITY_TO_AREA_FILE)):
        return
    df = pd.read_csv(PROV_CITY_TO_AREA_FILE)
    for index, row in df.iterrows():
        PROV_CITY_TO_AREA[(row['prov'], row['city'])] = row['area']
        PROV_CITY_TO_AREA_COUNT[(row['prov'], row['city'])] = row['count']


readProvCityToArea()


def setCounterToZero():
    """Set counter to zero"""
    global PROV_CITY_TO_AREA_COUNT
    for key in PROV_CITY_TO_AREA_COUNT.keys():
        PROV_CITY_TO_AREA_COUNT[key] = 0


def calcProvCityToAreaDF(write_to_file: bool = False):
    """Write province-city to area mapping to csv file"""
    global PROV_CITY_TO_AREA, PROV_CITY_TO_AREA_COUNT, PROV_CITY_TO_AREA_DF, CITY_COUNT_THRESHOLD
    rows = []
    for key, value in PROV_CITY_TO_AREA.items():
        if PROV_CITY_TO_AREA_COUNT[key] > CITY_COUNT_THRESHOLD:
            rows.append({'prov': key[0], 'city': key[1],
                         'area': value, 'count': PROV_CITY_TO_AREA_COUNT[key]})
    PROV_CITY_TO_AREA_DF = pd.DataFrame.from_records(rows)
    PROV_CITY_TO_AREA_DF.sort_values(
        by=['count'], ascending=False, inplace=True)
    if write_to_file:
        PROV_CITY_TO_AREA_DF.to_csv(PROV_CITY_TO_AREA_FILE, index=False)
    return PROV_CITY_TO_AREA_DF


def fill_row_area(row):
    """Fill area from PROV_CITY_TO_AREA dict"""
    global PROV_CITY_TO_AREA, gEmptyAreaCount
    if isNanOrNone(row['area']) and row['city'] != '':
        gEmptyAreaCount += 1
        return PROV_CITY_TO_AREA.get((row['prov'], row['city']), '')
    else:
        return row['area']


def dateToInt(date):
    """Convert date to int"""
    return int(datetime.strftime(date, '%Y%m%d'))


def read_data(
    scale: EstimateScale,
    col_list: list[str],
    date_span: int = 180,
    query: dict = {}
):
    """Read the data. Default date_span is 180 days"""
    if not isinstance(scale, EstimateScale):
        raise ValueError('scale must be an instance of EstimateScale')
    dateTo = dateToInt(scale.datePoint)
    dateFrom = dateToInt(scale.datePoint - timedelta(days=date_span))
    geoQuery = scale.get_geo_query()
    typeQuery = scale.get_type_query()
    saletpQuery = scale.get_saletp_query()
    filter = {
        'ptype': 'r',
        'onD': {
            '$gte': dateFrom,
            '$lte': dateTo}}
    filter = {**filter, **geoQuery, **typeQuery, **saletpQuery,  **query}
    return read_data_by_query(filter, col_list)


def read_data_by_query(
    query: dict,
    col_list: list[str],
    mongodb: MongoDB = None
):
    """ use mongo connection to read data from mongodb
    """
    if mongodb is None:
        mongodb = MongoDB()
    logger.info(f'Mongo Query: {str(query)[0:160]}')
    start_time = time.time()
    result = mongodb.load_data('properties', col_list, query)
    if BaseCfg.isDebug():
        logger.debug(f'columns: {result.columns} shape: {result.shape}')
        print_dateframe(result)
    end_time = time.time()
    logger.info(
        f'Result data shape:{result.shape}; used: {end_time - start_time}s')
    return result


def update_records(
    df: pd.DataFrame,
    col_list: list[str],
    db_col_list: list[str],
    id_index: int = 5,
    mongodb: MongoDB = None,
):
    if mongodb is None:
        mongodb = MongoDB()
    start_time = time.time()
    not_none_db_col_list = []
    to_save_col_list = []
    for i, col in enumerate(db_col_list):
        if col is not None:
            not_none_db_col_list.append(col)
            to_save_col_list.append(col_list[i])
    if len(not_none_db_col_list) == 0:
        logger.warning('No columns to save')
        return
    df = df[to_save_col_list].copy()
    df.columns = not_none_db_col_list
    data_to_save = df.to_dict(orient='index')
    savedCount = 0
    lastToSet = {}
    lastId = ''
    for key, value in data_to_save.items():
        id = key[id_index]
        # logger.debug(f'Updating key: {key} , id: {id}')
        # only update the columns that are numeric
        toSet = {}
        for k, v in value.items():
            if isinstance(v, (int, float)) and not isnan(v) and v != 0:
                toSet[k] = v
        if len(toSet) < 2:
            # for value and accuracy, we need to update at least 2 columns
            continue
        mongodb.updateOne(
            'properties',
            {'_id': id},
            {'$set': toSet}
        )
        lastId = id
        lastToSet = toSet
        savedCount += 1
    end_time = time.time()
    logger.info(
        f'Saved {savedCount}/{df.shape[0]} rows, used: {end_time - start_time}s lastToSet({lastId}):{lastToSet}')


class DataSource:
    """DataSource class to store data sources.
    Dataframes:
        df_raw: the raw data frame
        df_transformed: the transformed data frame
        df_grouped: the grouped data frame from transformed data frame

    Parameters:
    =================
    scale: EstimateScale
        the scale to read data from mongodb
    query: dict, optional
        the extra query combined to the scale to read data from mongodb
    col_list: list[str]
        the columns to read from mongodb
    """

    all_data_query = {
        # test: '_id': {'$in': ['TRBE5467506', 'TRBC5467511']},
        # only use data after 2021 for test purpose
        'onD': {'$gt': 20200101},
        'ptype': 'r',
        'prov': 'ON',
        'area': {
            '$in': [
                'Toronto', 'York', 'Halton', 'Durham',
                'Peel', 'Hamilton', 'Simcoe',
            ]
        },
    }
    all_data_col_list: list[str] = [
        '_id', 'onD', 'offD', 'sldd', 'lst', 'status',
        'prov', 'area', 'city', 'cmty', 'addr', 'uaddr',
        'st', 'st_num', 'lat', 'lng', 'unt', 'zip',
        'ptype', 'ptype2', 'saletp', 'pstyl', 'ptp',
        'lp', 'lpr', 'sp', 'tax', 'taxyr', 'mfee',
        'bdrms', 'tbdrms', 'br_plus', 'bthrms', 'kch', 'kch_plus',
        'bths', 'rms', 'bsmt', 'schools', 'zone',
        'gr', 'tgr', 'gatp',
        'depth', 'flt',
        'heat', 'feat', 'constr', 'balcony', 'ac',
        'den_fr', 'ens_lndry', 'fce', 'lkr',
        'sqft', 'rmSqft', 'bltYr', 'rmBltYr',
        'cac_inc', 'comel_inc', 'heat_inc', 'prkg_inc',
        'hydro_inc', 'water_inc', 'all_inc', 'pvt_ent',
        'insur_bldg', 'tv',
        'pets', 'laundry', 'laundry_lev',
        'daddr', 'commuId', 'park_fac',
        'comm', 'rltr', 'la', 'la2',
    ]

    def __init__(
        self,
        scale: Union[EstimateScale, list[EstimateScale]],
        query: dict = None,
        col_list: list[str] = None
    ):
        """Initialize DataSource object.
        """
        self.scale = scale
        self.query = query
        self.col_list = col_list if col_list else DataSource.all_data_col_list
        self.df_raw = None
        self.df_transformed = None
        self.df_grouped = None
        if isinstance(scale, list):
            # or condition for scale tuple
            or_query = []
            for s in scale:
                or_query.append(s.get_query())
            self._query = {**query, '$or': or_query}
        else:
            self._query = {**query, **self.scale.get_query()}

    def __str__(self):
        return f'DataSource: {self._query}'

    def get_query(self):
        return self._query

    def load_raw_data(self):
        """Load raw data from mongodb"""
        self.df_raw = read_data_by_query(
            self._query, self.col_list)
        self._build_prov_city_to_area()
        self._fill_df_raw_area()
        self._build_prov_city_to_area(True)  # rebuild map and write to file
        self._build_scale_tree()

    def _fill_df_raw_area(self):
        """Fill area column in df_raw"""
        global PROV_CITY_TO_AREA, gEmptyAreaCount
        self.df_raw['area'] = self.df_raw.apply(fill_row_area, axis=1)
        logger.info(f'Empty area rows: {gEmptyAreaCount}')

    def _build_prov_city_to_area(self, write_to_file=False):
        """Build PROV_CITY_TO_AREA dict from df_raw"""
        global PROV_CITY_TO_AREA, PROV_CITY_TO_AREA_COUNT
        # reset global dict count to empty
        setCounterToZero()
        for prov, area, city in self.df_raw[['prov', 'area', 'city']].values:
            if isinstance(area, str) and area != '' and isinstance(city, str) and city != '' and isinstance(prov, str) and prov != '':
                if PROV_CITY_TO_AREA_COUNT.get((prov, city), 0) == 0:
                    PROV_CITY_TO_AREA[(prov, city)] = area
                    PROV_CITY_TO_AREA_COUNT[(prov, city)] = 1
                else:  # not exist
                    if PROV_CITY_TO_AREA[(prov, city)] == area:
                        PROV_CITY_TO_AREA_COUNT[(prov, city)] += 1
                        continue
                    # conflict
                    change = False
                    areaBeforeChange = PROV_CITY_TO_AREA[(prov, city)]
                    if PROV_CITY_TO_AREA_COUNT[(prov, city)] < 20:
                        # change to new area if count < 10
                        PROV_CITY_TO_AREA[(prov, city)] = area
                        # reset count to 1
                        PROV_CITY_TO_AREA_COUNT[(prov, city)] = 1
                        change = True
                    if PROV_CITY_TO_AREA_COUNT[(prov, city)] < 100:
                        # only log conflict if conflict count < 100
                        logger.warning(
                            f'PROV_CITY_TO_AREA conflict: {(prov, city)}: {areaBeforeChange} {PROV_CITY_TO_AREA_COUNT[(prov, city)]} vs New:{area}. {"Change to New" if change else "Keep old value"}')

        calcProvCityToAreaDF(write_to_file=write_to_file)
        return PROV_CITY_TO_AREA

    def load_df_grouped(
        self,
        id_list: list[str] = None,
        preprocessor: Preprocessor = None,
    ) -> pd.DataFrame:
        """Load new data from mongodb.
        This function is used to load new data from mongodb for prediction.
        """
        global PROV_CITY_TO_AREA
        query = {'_id': {'$in': id_list}}
        idCount = len(id_list)
        logger.info(f'query ids: {idCount}')
        if idCount > 500000:  # 500K, query has a limit of 16MB
            # split query
            df_raws = []
            while idCount > 0:
                query['_id']['$in'] = id_list[:500000]
                df_raws.append(read_data_by_query(query, self.col_list))
                id_list = id_list[500000:]
                idCount = len(id_list)
            df_raw_to_predict = pd.concat(df_raws)
        else:
            df_raw_to_predict = read_data_by_query(query, self.col_list)
        # fill missing area
        df_raw_to_predict['area'] = df_raw_to_predict.apply(
            lambda row: PROV_CITY_TO_AREA.get((row['prov'], row['city']), 'Other'), axis=1)
        logger.debug(PROV_CITY_TO_AREA)
        logger.debug(df_raw_to_predict[['prov', 'area', 'city']])
        df_transformed_to_predict = preprocessor.transform(df_raw_to_predict)
        # groupby and reindex by EstimateScale
        df_grouped_to_predict = df_transformed_to_predict.set_index([
            'saletp-b', 'ptype2-l',
            'prov', 'area', 'city',
            '_id',
        ]).sort_index(
            level=[0, 1, 2, 3, 4],
            ascending=[True, True, True, True, True],
            inplace=False,
        )
        return df_grouped_to_predict

    def concat_df_grouped(self, df_grouped: pd.DataFrame):
        """Concat grouped data with new data.
        """
        self.df_grouped = pd.concat(
            [self.df_grouped, df_grouped], axis=0)

    def getLeafScales(
        self,
        propType: str = None,
        sale: bool = None,
    ):
        if isinstance(self.scale, list):
            leafScales = []
            for s in self.scale:
                leafScales.extend(s.getLeafScales(
                    propType=propType, sale=sale))
            return leafScales
        else:
            return self.scale.getLeafScales(propType=propType, sale=sale)

    def _build_scale_tree(self):
        """Build scale tree from raw data."""
        logger.debug(
            f'Build Scale Tree.')
        if isinstance(self.scale, list):
            for s in self.scale:
                s.buildAllSubScales(PROV_CITY_TO_AREA_DF)
        else:
            self.scale.buildAllSubScales(PROV_CITY_TO_AREA_DF)

    def transform_data(self, preprocessor: Preprocessor = None, clear_data: bool = True):
        """ Transform data with preprocessor then group them.
        If raw data is not loaded, load it first.
        Args:
            preprocessor (Preprocessor, optional): Defaults to None.
                provide fit_transform method to transform df_raw to df_transformed.

        Returns:
            DataFrame: transformed DataFrame.
        """
        if self.df_raw is None:
            self.load_raw_data()
        self.df_transformed = preprocessor.fit_transform(self.df_raw)
        self.flag_to_include_else = preprocessor.flag_to_include_else
        # groupby and reindex by EstimateScale
        self.df_grouped = self.df_transformed.set_index([
            'saletp-b', 'ptype2-l',
            'prov', 'area', 'city',
            '_id',
        ]).sort_index(
            level=[0, 1, 2, 3, 4],
            ascending=[True, True, True, True, True],
            inplace=False,
        )
        if clear_data:
            self.df_raw = None
            self.df_transformed = None
        return self.df_grouped

    # @debug
    def get_df(
        self,
        scale: EstimateScale,
        cols: list[str] = None,
        date_span: int = 180,
        need_raw: bool = False,
        suffix_list: list[str] = None,
        copy: bool = False,
        sample_size: int = None,
        filter_func: (pd.Series) = None,
        numeric_columns_only: bool = False,
        prefer_estimated: bool = False,
        df_grouped: pd.DataFrame = None,
        ad_cols: bool = False
    ):
        """Get dataframe from stored data.
        """
        if date_span is None:
            date_span = 180
        if suffix_list is None:
            suffix_list = ['-b', '-n', '-c']
        slices = []
        # filter data by scale and date_span
        # saletp_b
        if scale.sale is True:
            slices.append(slice(0, 0))
        elif scale.sale is False:
            slices.append(slice(1, 1))
        else:  # scale.sale is None
            slices.append(slice(None))
        # ptype2_l
        if scale.propType is not None:
            slices.append(slice(scale.propType, scale.propType))
        else:
            slices.append(slice(None))
        # prov, area, city
        if scale.prov is not None:
            slices.append(slice(scale.prov, scale.prov))
        else:
            slices.append(slice(None))
        if scale.area is not None:
            slices.append(slice(scale.area, scale.area))
        else:
            slices.append(slice(None))
        if scale.city is not None:
            slices.append(slice(scale.city, scale.city))
        else:
            slices.append(slice(None))
        slices.append(slice(None))
        if df_grouped is None: # yes, its None and we use df.grouped (final data frame)
            df_grouped = self.df_grouped
        
        #df_grouped.to_excel("what's_popin000.xlsx")
        
        #rd = df_grouped
        #print("!!!!!!!!")
        #print(tuple(slices))
        #print(df_grouped.shape)
        rd = df_grouped.loc[tuple(slices), :]
        #print(rd.shape)
                
        logger.debug(f'{slices} {len(df_grouped.index)}=>{len(rd.index)}')
        
        if len(rd.index) == 0:
            return None
        
        # onD:
            
        #rd.to_excel("what's_popin00.xlsx")
        
        
        if date_span > 0: 
            rd = rd.loc[rd.onD.between(
                dateToInt(scale.datePoint - timedelta(days=date_span)),
                dateToInt(scale.datePoint)
            )]
        
        logger.debug(
            f'{scale.datePoint-timedelta(days=date_span)}-{scale.datePoint} {len(rd.index)}')
        
        
        if rd.shape[0] == 0:
            print("FUCKCKCKCKCK")
            exit()
        
        #rd.to_excel("what's_popin0.xlsx")
        
        # filter data by filter_func
        if filter_func is not None:
            bol = rd.apply(filter_func, axis=1, result_type='reduce')
            rd = rd.loc[bol]
        
        logger.debug(f'after filter_func {len(rd.index)}')
  
        if len(rd.index) == 0:
            logger.debug('index is empty')
            return None
        # sample data
        if sample_size is not None and sample_size < rd.shape[0]:
            rd = rd.sample(n=sample_size, random_state=1)
            
        #rd.to_excel("what's_popin1.xlsx")
        
        # select columns from cols
        existing_cols = rd.columns.tolist()
        
        if cols is not None:
            columns = []
            for col in cols:
                found = False
                # {zip}, {zip-n} {zip,zip-n}
                # ['-b', '-n', '-c', '-l', ]:  '-l' is only in 'ptype2-l'
                for suffix in suffix_list:
                    for c in existing_cols:
                        if c.startswith(col) and c.endswith(suffix):
                            columns.append(c)
                            # logger.debug(f'column[{col}] found as [{c}]')
                            found = True
                if (col in existing_cols) and (need_raw or not found):
                    columns.append(col)
                    # logger.debug(f'column[{col}] found as [{c}] (raw)')
                    found = True
                if not found:
                    logger.debug(f'column[{col}] not found')
            columns = list(dict.fromkeys(columns))  # remove duplicates
        else:
            columns = existing_cols
            
        # ad_cols is refering to the self.encoded_hot in preposcesor.py 
        
        if ad_cols: # should boolean value
            one = set(columns) # derived fearue from default
            two = set(cols) # the default features
            three = set(existing_cols)
            u = one.union(two)
            dif = list(three.difference(u))
            
            columns = columns + dif # subject to change!
        
        rd = rd.loc[:, columns]
        #rd.to_excel("what's_popin333.xlsx")
        
        # if numeric_columns_only is True, only keep numeric columns
        if numeric_columns_only:
            numeric_columns = self.get_numeric_columns(
                df=rd, prefer_estimated=prefer_estimated)
            rd = rd.loc[:, numeric_columns]
        # return dataframe or copy
        #rd.to_excel("what's_popin2.xlsx")
        return rd.copy() if copy else rd

    def get_numeric_columns(
        self,
        df: pd.DataFrame,
        exclude_columns: list[str] = None,
        prefer_estimated: bool = False,
    ) -> list[str]:
        """Get the numeric columns

        Parameters
        ==========
        df: pd.DataFrame
        exclude_columns: list[str]
            columns to be excluded from the result
        prefer_estimated_value: bool
            if True, return the estimated value column if it exists,
            and remove the original column.
        """
        if exclude_columns is None:
            exclude_columns = []
        numeric_columns = []
        for col in df.columns:
            try:
                if (col not in exclude_columns) and \
                    (df[col].dtype == 'float64' or
                     df[col].dtype == 'int64'):
                    numeric_columns.append(col)
            except Exception as e:
                self.logger.error(
                    f'Error in getting numeric column:{col} error:{e}')
        if prefer_estimated:
            new_cols = numeric_columns.copy()
            for col in numeric_columns:
                if col.endswith('-e'):
                    for suffix in ['-c', '-n', '-b']:
                        orig_col = col[:-2] + suffix
                        if orig_col in numeric_columns:
                            new_cols.remove(orig_col)
            numeric_columns = new_cols
        return numeric_columns

    def writeback(
        self,
        col: Union[str, list[str]],
        y: Union[pd.Series, pd.DataFrame],
        df_grouped: pd.DataFrame = None,
        db_col: Union[str, list[str]] = None,
    ) -> None:
        """write y to df_grouped
        """
        if df_grouped is None:
            df_grouped = self.df_grouped
        yIsSeries = False
        if isinstance(y, pd.Series):
            yIsSeries = True
        if isinstance(col, str):
            col = [col]
        if not isinstance(col, list):
            raise Exception(
                f'col must be str or list[str], but got {type(col)}')
        # self.y = y  # for debug
        for c in col:
            if c not in df_grouped.columns:
                #df_grouped.loc[:, c] = NaN
                df_grouped[c] = NaN
            if yIsSeries:
                y.name = c
                df_grouped.loc[y.index, c] = y
            else:
                # df_grouped.loc[y.index, c] = y.loc[y.index, c]
                df_grouped.update(y, join='left', overwrite=True)
        if db_col is not None:
            if isinstance(db_col, str):
                db_col = [db_col]
            elif isinstance(db_col, list):
                if len(db_col) != len(col):
                    raise Exception(
                        f'len(db_col) must be equal to len(col), but got {len(db_col)}({db_col}) and {len(col)}({col})')
            else:
                raise Exception(
                    f'db_col must be str or list[str], but got {type(db_col)}')
            update_records(df_grouped, col_list=col,
                           db_col_list=db_col, id_index=5)
        return df_grouped


class TrendDataSource:
    """TODO: Trend Data Source.
    Provide dataframe for trend analysis.
    Columns:
    indexes:
        saletp-b, ptype2-l, prov, area, city, periodId
    keys:
        year, month, week,
    first level:
        new, sold, off, pureNew, pc,
        askPriceAvg, soldPriceAvg,
        soldDomAvg, offDomAvg,
        askPerTaxAvg, soldPerTaxAvg,
        startAskPriceAvg, endAskPriceAvg,
        startAvailAvg, endAvailAvg,
    secondary level:
        pureNewPerNew, pureNewPerSold, pureNewPerOff,
        soldPerNew, soldPerOff,
        soldDomAvg, soldDomMedian,
        offDomAvg, offDomMedian,
        startAvgPrice, endAvgPrice, priceDiffAvg,
        startMedianPrice, endMedianPrice, priceDiffMedian,
        pcPerNewAvg, priceChangePerNewMedian,
        priceSoldPerAsk, priceChangedPercent, priceChanged,
    date features:
        periodWeek, periodMonth, periodQuarter, periodYear,

    """

    def __init__(self) -> None:
        pass
