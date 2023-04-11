import numbers
import re
from .base_cfg import BaseCfg
import pandas as pd
import numpy as np
from math import isnan
from datetime import datetime, timezone
import psutil

UTC_OFFSET_TIMEDELTA = datetime.utcnow() - datetime.now()

logger = BaseCfg.getLogger(__name__)


def selectFromTwo(a, b):
    if (a is not None) and (not isnan(a)):
        return a
    return b


def utc_to_local(utc_dt):
    return utc_dt.replace(tzinfo=timezone.utc).astimezone(tz=None)


def local_to_utc(local_dt):
    return local_dt + UTC_OFFSET_TIMEDELTA


def get_utc_datetime_from_str(str_datetime):
    if len(str_datetime) == 10:
        str_datetime = str_datetime + ' 00:00:00'
    return datetime.strptime(str_datetime, '%Y-%m-%d %H:%M:%S')


def print_dateframe(df):
    # more options can be specified also
    with pd.option_context('display.max_rows', BaseCfg.pd_max_rows, 'display.max_columns', BaseCfg.pd_max_columns):
        print(df)


def index_min(values):
    # min(range(len(values)), key=values.__getitem__)
    return np.argmin(values)


def print_all_columns(df):
    pd.set_option('display.max_columns', None)
    print(df.columns.values, df.columns.values.shape)


def allTypeToFloat(value):
    v_type = type(value)
    if v_type is int:
        return float(value)
    if v_type is float:
        if isnan(value):
            return None
        return value
    if v_type is str:
        num_array = re.findall(r'\d+(?:\.\d+)?', value)
        value = ''.join(num_array)
        if len(value) > 0:
            if value.isnumeric():
                return float(value)
            return float(num_array[0])
    if v_type is list:
        return allTypeToFloat(value[0])
    return None


def allTypeToInt(value):
    v_type = type(value)
    if v_type is int:
        return value
    if v_type is float:
        if isnan(value):
            return None
        return int(value)
    if v_type is str:
        int_array = re.findall(r'\d+(?:\.\d+)?', value)
        value = ''.join(int_array)
        if len(value) > 0:
            return int(float(value))
    if v_type is list:
        return allTypeToInt(value[0])
    return None


reAllNumber = re.compile(r'^\s*\d+(?:\.\d+)?\s*$')


def stringToFloat(value):
    if isinstance(value, str) and reAllNumber.match(value):
        return float(value)
    if (isinstance(value, float) or isinstance(value, int)) and not isnan(value):
        return float(value)
    return None


def stringToInt(value):
    if isinstance(value, str) and reAllNumber.match(value):
        return int(value)
    if (isinstance(value, float) or isinstance(value, int)) and not isnan(value):
        return int(value)
    return None


def ymdFromNum(dayNum):
    if isnan(dayNum):
        return None, None, None
    dayNum = int(dayNum)
    day = dayNum % 100
    year = dayNum // 10000
    month = (dayNum // 100) - (year * 100)
    if month < 1:
        month = 1
        print(f'month < 1 : {dayNum}')
    if month > 12:
        if dayNum < 2000001:  # there are numbers like 2017125
            year = dayNum // 1000
            month = (dayNum % 1000) // 100
        if month > 12:
            month = 12
            print(f'month > 12 : {dayNum}')
    return year, month, day


def dateFromNum(dayNum):
    year, month, day = ymdFromNum(dayNum)
    if year is None:
        return None
    return datetime(year, month, day, 0, 0)


def dateFromNumOrNow(dayNum):
    if isnan(dayNum):
        return datetime.now()
    return dateFromNum(dayNum)


def daysOfDifferenceFromNumRough(date1: int, date2: int) -> int:
    y1, m1, d1 = ymdFromNum(date1)
    y2, m2, d2 = ymdFromNum(date2)
    return (y1-y2)*365 + (m1-m2)*30 + (d1-d2)


def daysOfDifference(date1: datetime, date2: datetime) -> int:
    return (date1 - date2).days


def dateToNum(date: datetime = datetime.today()):
    return date.year * 10000 + date.month * 100 + date.day


def isNanOrNone(value):
    if value is None:
        return True
    if (value == 'nan') or (value == ''):
        return True
    if isinstance(value, numbers.Number) and isnan(value):
        return True
    return False


def getUniqueLabels(X: pd.Series) -> list:
    return X.unique().tolist()


def debug(fn):
    def wrapper(*args, **kwargs):
        logger.debug(f"Invoking {fn.__name__}")
        logger.debug(f"  args: {args}")
        logger.debug(f"  kwargs: {kwargs}")
        result = fn(*args, **kwargs)
        logger.debug(f"  returned {result}")
        return result
    return wrapper


def columnValues(df: pd.DataFrame, col: str) -> list:
    return df[col].unique().tolist()


def getRoundFunction(n=1, min=0, max=1000000, outRangeValue=0):
    def roundByN(value):
        nonlocal n, min, max, outRangeValue
        if isnan(value):
            return value
        retValue = int(round(value / n) * n)
        if retValue < min or retValue > max:
            return outRangeValue
        return retValue
    return roundByN


def roundByN(value, n=1):
    if isnan(value):
        return value
    return int(round(value / n) * n)


def expendList(list1, list2):
    addedCounter = 0
    for item in list2:
        if (item is not None) and (item not in list1):
            list1.append(item)
            addedCounter += 1
    return addedCounter


def flattenList(lists):
    return [item for sublist in lists for item in sublist]


def printColumns(df, start: str = None):
    cols = []
    for c in df.columns:
        if start is not None:
            if c.startswith(start):
                cols.append(c)
        else:
            cols.append(c)
    cols.sort()
    for c in cols:
        print(f'{c} {df[c].dtype}')


def getMemoryUsage():
    # Getting all memory using os.popen()
    mem = psutil.virtual_memory()
    # Memory usage
    print(mem)
    return mem[0], mem[3], mem[4]


def getMemoryLimitedExtraProcessNumber():
    total_memory, used_memory, free_memory = getMemoryUsage()
    return int(free_memory / used_memory)


def addColumns(df: pd.DataFrame, colNames: list[str], value=None):
    new_columns = pd.DataFrame(
        {col: value for col in colNames}, index=df.index)
    return pd.concat([df, new_columns], axis=1)


def logDataframeChange(df1: pd.DataFrame, df2: pd.DataFrame, logger, message: str = ''):
    rowsBefore = len(df1.index)
    colsBefore = df1.shape[1]
    rowsAfter = len(df2.index)
    colsAfter = df2.shape[1]
    logger.info(
        f'{message} Rows dropped: {rowsBefore-rowsAfter}/{rowsBefore}=>{rowsAfter} \
Cols dropped: {colsBefore-colsAfter}/{colsBefore}=>{colsAfter}')
