
import datetime
from enum import Enum
import re
from typing import Union
from base.base_cfg import BaseCfg
import pandas as pd
from base.util import columnValues

# from regex import F, P
# from sympy import Q
logger = BaseCfg.getLogger(__name__)


class PropertyType:
    DETACHED = 'Detached'
    SEMI_DETACHED = 'Semi-Detached'
    TOWNHOUSE = 'Townhouse'
    CONDO = 'Condo'
    OTHER = 'Other'


class PropertyTypeRegexp:
    DETACHED = re.compile(
        '^Det|^Detached|Cottage|^Det Condo|Det W/Com Elements|Link|Rural Resid', re.IGNORECASE)
    SEMI_DETACHED = re.compile(
        '^Semi-|Semi-Detached|Semi-Det Condo', re.IGNORECASE)
    TOWNHOUSE = re.compile(
        '^Att|Townhouse|Townhouse Condo|Att/Row/Twnhouse', re.IGNORECASE)
    CONDO = re.compile(
        'Apartment|Condo Apt|Co-Op Apt|Co-Ownership Apt|Comm Element Condo|Leasehold Condo', re.IGNORECASE)


class EstimateScale:
    """Scale of Estimation

    Parameters
    ==========
    datePoint: datetime
    propType: str. Property type. One of PropertyType
    prov: str. Province
    area: str. Area
    city: str. City
    sale: bool. Sale or rent. None for both
    meta: dict. Meta data for the scale.
        Meta can be used to hold parameters, predictors, etc.
    """

    def __init__(
            self,
            datePoint: datetime,
            propType: str = None,
            prov: str = 'ON',
            area: str = None,
            city: str = None,
            sale: Union[bool, str] = None,
            meta: dict = None,
    ):
        if meta is None:
            meta = {}
        self.datePoint = datePoint
        self.propType = propType
        self.prov = prov
        self.area = area
        self.city = city
        self.sale = sale
        self.meta = meta  # meta data for the scale

    @classmethod
    def fromRepr(cls, repr: str):
        """Create an instance from a string representation."""
        propType, prov, area, city, sale, datePoint = repr.split('.')
        if propType == 'None' or propType == 'N':
            propType = None
        elif propType == 'DETACHED':
            propType = PropertyType.DETACHED
        elif propType == 'SEMI_DETACHED':
            propType = PropertyType.SEMI_DETACHED
        elif propType == 'TOWNHOUSE':
            propType = PropertyType.TOWNHOUSE
        elif propType == 'CONDO':
            propType = PropertyType.CONDO
        else:
            propType = PropertyType.OTHER
        if prov == '-':
            prov = None
        if area == '-':
            area = None
        if city == '-':
            city = None
        if (sale == 'BothSaleRent') or (sale == '-') or (sale == 'Both'):
            sale = None
        elif sale == 'Sale' or sale == 'S':
            sale = True
        elif sale == 'Rent' or sale == 'R':
            sale = False
        return cls(
            datePoint=datetime.datetime.strptime(datePoint, '%Y-%m-%d'),
            propType=propType,
            prov=prov,
            area=area,
            city=city,
            sale=sale,
        )

    def getSubScales(
        self,
        df_prov_area_city: pd.DataFrame = None,
    ) -> list:
        if hasattr(self, 'subScales'):
            return self.subScales
        self.subScales = []
        if self.propType is None:
            for propType in [PropertyType.DETACHED, PropertyType.SEMI_DETACHED, PropertyType.TOWNHOUSE, PropertyType.CONDO]:
                self.subScales.append(self.copy(propType=propType))
        elif self.prov is None and df_prov_area_city is not None:
            for prov in columnValues(df_prov_area_city, 'prov'):
                self.subScales.append(self.copy(prov=prov))
        elif self.area is None and df_prov_area_city is not None:
            for area in columnValues(df_prov_area_city[df_prov_area_city['prov'] == self.prov], 'area'):
                self.subScales.append(self.copy(area=area))
        elif self.city is None and df_prov_area_city is not None:
            for city in columnValues(df_prov_area_city[df_prov_area_city['prov'] == self.prov][df_prov_area_city['area'] == self.area], 'city'):
                self.subScales.append(self.copy(city=city))
        elif self.sale is None:
            for sale in [True, False]:
                self.subScales.append(self.copy(sale=sale))
        else:
            self.subScales = None
            return None
        return self.subScales

    def buildAllSubScales(
        self,
        df_prov_area_city: pd.DataFrame = None,
    ):
        subs = self.getSubScales(df_prov_area_city=df_prov_area_city)
        if subs is None:
            return
        for sub in subs:
            sub.buildAllSubScales(df_prov_area_city=df_prov_area_city)

    def getLeafScales(
        self,
        propType: str = None,
        sale: bool = None,
    ):
        """Get all leaf scales from the scale tree.
        Parameters:
        -----------
        propType: str, optional (default=None)
            property type to filter scales
        sale: bool, optional (default=None)
            sale type to filter scales
        """
        if getattr(self, 'subScales', None) is None:
            if (propType is None or self.propType == propType) and (sale is None or self.sale == sale):
                return [self]
            else:
                return []
        leafScales = []
        for sub in self.subScales:
            leafScales.extend(sub.getLeafScales(propType, sale))
        return leafScales

    def copy(
        self,
        datePoint: datetime = None,
        propType: str = None,
        prov: str = None,
        area: str = None,
        city: str = None,
        sale: bool = None,
    ):
        sale = sale if sale is not None else self.sale
        return EstimateScale(
            datePoint=datePoint or self.datePoint,
            propType=propType or self.propType,
            prov=prov or self.prov,
            area=area or self.area,
            city=city or self.city,
            sale=sale,
        )

    def __repr__(self):
        keys = []
        if self.propType is None:
            keys.append('N')
        elif self.propType is PropertyType.DETACHED:
            keys.append('D')
        elif self.propType is PropertyType.SEMI_DETACHED:
            keys.append('S')
        elif self.propType is PropertyType.TOWNHOUSE:
            keys.append('T')
        elif self.propType is PropertyType.CONDO:
            keys.append('C')
        else:
            keys.append('O')
        for key in ['prov', 'area', 'city']:
            if self.__dict__[key]:
                keys.append(str(self.__dict__[key]))
            else:
                keys.append('-')
        if self.sale:
            keys.append('S')
        elif self.sale is False:
            keys.append('R')
        else:
            keys.append('-')
        keys.append(self.datePoint.strftime('%Y-%m-%d'))
        return '.'.join(keys)

    def __eq__(self, __o: object) -> bool:
        return repr(self) == repr(__o)

    def __str__(self):
        keys = []
        if self.propType is None:
            keys.append('AllType')
        elif self.propType is PropertyType.DETACHED:
            keys.append('Detached')
        elif self.propType is PropertyType.SEMI_DETACHED:
            keys.append('Semidetached')
        elif self.propType is PropertyType.TOWNHOUSE:
            keys.append('Townhouse')
        elif self.propType is PropertyType.CONDO:
            keys.append('Condo')
        else:
            keys.append('Other')
        for key in ['prov', 'area', 'city']:
            if self.__dict__[key]:
                keys.append(str(self.__dict__[key]))
        if self.sale is True:
            keys.append('Sale')
        elif self.sale is False:
            keys.append('Rent')
        else:
            keys.append('BothSaleRent')
        keys.append(self.datePoint.strftime('%Y-%m-%d'))
        return '.'.join(keys)

    def get_type_query(self):
        """Get the query for property type"""
        if self.propType is None:
            query = {}
        elif self.propType is PropertyType.DETACHED:
            query = {'ptype2': PropertyTypeRegexp.DETACHED}
        elif self.propType is PropertyType.SEMI_DETACHED:
            query = {'ptype2': PropertyTypeRegexp.SEMI_DETACHED}
        elif self.propType is PropertyType.TOWNHOUSE:
            query = {'ptype2': PropertyTypeRegexp.TOWNHOUSE}
        elif self.propType is PropertyType.CONDO:
            query = {'ptype2': PropertyTypeRegexp.CONDO}
        return query

    def get_geo_query(self):
        query = {}
        if self.prov:
            query['prov'] = self.prov
        if self.area:
            query['area'] = self.area
        if self.city:
            query['city'] = self.city
        return query

    def get_saletp_query(self):
        query = {}
        if self.sale is True:
            query['saletp'] = 'Sale'
        elif self.sale is False:
            query['saletp'] = 'Rent'
        return query

    def get_query(self):
        query = {'ptype': 'r'}
        query.update(self.get_type_query())
        query.update(self.get_geo_query())
        query.update(self.get_saletp_query())
        return query
