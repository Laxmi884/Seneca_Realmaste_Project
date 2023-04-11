# top level class for training models and predicting results
# Run 3 modes:
#  1. initialize transformers.
#    - real all data and build label encoder database
#    - save encoder database
#    - hold data in memory
#  2. train models.
#    - load encoder database if not yet
#    - train and writeback predictors
#    - train models
#    - save models
#  3. predict results.
#    - load encoder database if not yet
#    - load models
#    - predict results
import os

import socket
import sys
import time
from pandas import DataFrame
from base.timer import Timer
from base.util import dateFromNum
import numpy as np
import pandas as pd
from base.base_cfg import BaseCfg
from base.const import CONCURRENT_PROCESSES_MAX, DEFAULT_CITY, DEFAULT_DATE_POINT_DATE, DEFAULT_PROV, DEFAULT_START_DATA_DATE, PROPERTIES_COLLECTION
from base.mongo import MongoDB
from base.model_gridfs_store import GridfsStore
from base.sysDataHelpers import setSysdataTs
from data.data_source import DataSource
from data.estimate_scale import EstimateScale, PropertyType
from transformer.preprocessor import Preprocessor
import logging


from estimator.builtyear_lgbm_estimate_manager import BuiltYearLgbmManager
from estimator.sqft_lgbm_estimate_manager import SqftLgbmEstimateManager
from estimator.rent_lgbm_estimate_manager import RentLgbmManager
from estimator.value_lgbm_estimate_manager import ValueLgbmEstimateManager
from estimator.soldprice_lgbm_estimate_manager import SoldPriceLgbmEstimateManager
from signal import Signals, signal, SIGINT
from sys import exit

SYSDATA_COLLECTION = 'sysdata'
WATCH_FIELDS = ('addr', 'lat', 'lp', 'lpr', 'city', 'onD',
                'bdrms', 'br_plus', 'bthrms', 'kch', 'kch_plus', 'gr', 'ptype2', 'pstyl',
                'sqft', 'age', 'rmSqft', 'rmBltYr',)

logger = BaseCfg.getLogger(__name__)


class Organizer:
    """Organizer class for training and predicting models."""

    def __init__(self, use_baseline: bool = False, clear_data: bool = True):
        """Initialize the class."""
        self.use_baseline = use_baseline
        self.clear_data = clear_data
        # connect database / check file system
        self.mongodb = MongoDB()
        self.raw_data: DataFrame = None
        self.prep_data: DataFrame = None
        self.data_source: DataSource = None
        self.model_store = GridfsStore(
            # self.mongodb.getDb(),
            collection='ml_fs',
            prefix='ml_',
        )
        self.root_preprocessor: Preprocessor = None
        self.writeback_predictors = []
        self.predictors = []
        self.default_all_scale = EstimateScale(
            datePoint=dateFromNum(DEFAULT_DATE_POINT_DATE),
            propType=None,  # PropertyType.DETACHED,
            prov=DEFAULT_PROV,
            area=None,
            city=DEFAULT_CITY,
            sale=None,
        )
        self.default_sale_scale = EstimateScale(
            datePoint=dateFromNum(DEFAULT_DATE_POINT_DATE),
            propType=PropertyType.SEMI_DETACHED,
            prov='ON',
            area='Toronto',
            city='Toronto',
            sale=True,
        )
        self.default_rent_scale = EstimateScale(
            datePoint=dateFromNum(DEFAULT_DATE_POINT_DATE),
            propType=PropertyType.DETACHED,
            prov='ON',
            area='Toronto',
            city='Toronto',
            sale=False,
        )
        self.num_procs = CONCURRENT_PROCESSES_MAX
        self.estimate_managers = {}
        # self.similar_properties = {'on': {}, 'off': {}}
        # self.timed_values = {'sale':{}, 'rent':{}}
        self.__update_status('program', 'run')

    def __update_status(self, job, status):
        self.__status = status
        logger.info(f'Organizer job({job}) status: {status}')
        setSysdataTs(self.mongodb, SYSDATA_COLLECTION, {
            'id': f"prop.Organizer_{socket.gethostname()}",
            'batchNm': sys.argv[0],
            'job': job,
            'status': status,
            'mt': BaseCfg.getTs()
        })

    def end(self):
        """Quit the program."""
        self.__update_status('program', 'done')
        self.mongodb.close()
        logger.info('Organizer end')

    def load_data(self, scale: EstimateScale = None):
        """Load data from database."""
        self.__update_status('load data', 'run')
        query = {
            # '_id': {'$in': ['TRBW474049', 'TRBW4874697']},
            'onD': {'$gt': DEFAULT_START_DATA_DATE},
        }
        if scale is None:
            scale = self.default_all_scale
        # area search does not work well, since many records have no area
        # for area in ['York', 'Peel', 'Toronto', 'Durham']:
        #     # ,'Hamilton','Waterloo','Niagara','Ottawa']:
        #     scales.append(self.default_all_scale.copy(area=area))
        self.data_source = DataSource(
            scale=scale,
            query=query)
        self.data_source.load_raw_data()
        self.__update_status('load data', 'done')

    def init_transformers(self):
        """Initialize the internal structures for transform and hold data in memory."""
        self.__update_status('init transformers', 'run')
        if self.data_source is None:
            self.load_data()
        self.root_preprocessor = Preprocessor(use_baseline=self.use_baseline)
        self.data_source.transform_data(
            self.root_preprocessor, self.clear_data)
        self.__update_status('init transformers', 'done')

    def train_models(self):
        for estimate_manager in [
            BuiltYearLgbmManager,
            SqftLgbmEstimateManager,
            RentLgbmManager,
            ValueLgbmEstimateManager,
            SoldPriceLgbmEstimateManager,
        ]:
            em = estimate_manager(self.data_source) # add cols
            em.load_scales()
            em.train()
            if em.name not in self.estimate_managers:
                self.estimate_managers[em.name] = {}
            self.estimate_managers[em.name][em.model_name] = em
            logger.info(f"trained {em.name} {em.model_name}")
            if hasattr(em, 'writeback'):
                em.writeback()
                logger.info(f"writeback {em.name} {em.model_name}")

    def save_models(self):
        for estimate_manager in self.estimate_managers.values():
            for em in estimate_manager.values():
                em.save(self.model_store)

    def load_models(self):
        """load models from database."""
        # TODO: load models from database
        pass

    def predict(
        self,
        id_list: list[str],
        writeback: bool = False
    ) -> tuple[pd.DataFrame, list[str], list[str]]:
        """Predict the values for the given list of ids."""
        self.__update_status('predict', 'run')
        if self.data_source is None:
            self.load_data()
        if self.root_preprocessor is None:
            self.init_transformers()
        if not self.estimate_managers:
            self.load_models()
        df_grouped_result, added_cols, db_cols = self.__predict(
            id_list, writeback)
        # self.__predict_parallel(id_list)
        self.__update_status('predict', 'done')
        return df_grouped_result, added_cols, db_cols

    def __predict(
        self,
        id_list: list[str],
        writeback: bool = False
    ) -> tuple[pd.DataFrame, list[str]]:
        """Predict the values for the given list of ids."""
        df_grouped = self.data_source.load_df_grouped(
            id_list, self.root_preprocessor)
        added_cols = []
        db_cols = []
        for estimate_manager in self.estimate_managers.values():
            # estimater type lavel
            for em in estimate_manager.values():
                # model level
                logger.info(
                    f'-------------predict {em.name} {em.model_name} {df_grouped.shape[0]}----------------------------')
                df_y, y_col_names, y_db_col_names = em.estimate(df_grouped)
                if df_y is None:
                    continue
                logger.info(df_y)
                logger.info(
                    f'-----------------predict done: {em.name} {em.model_name} {df_y.shape[0]}/{df_grouped.shape[0]}------------------------')
                # merge the prediction to the original data
                # df_grouped = pd.merge(
                #     df_grouped, df_y, how='left', left_index=True, right_index=True)
                df_grouped = self.data_source.writeback(
                    col=y_col_names,
                    y=df_y,
                    df_grouped=df_grouped,
                    db_col=y_db_col_names,
                )
                added_cols.extend(y_col_names)
                db_cols.extend(y_db_col_names)
        return df_grouped, added_cols, db_cols

    def watch_n_predicting(self, resume: bool = False):
        """Watch and predicting."""
        self.__update_status('watch', 'run')
        #query = {'ptype': 'r', 'status': 'A'}
        query = self.data_source.get_query()

        def save_resume_token(resume_token):
            self.mongodb.updateOne(
                SYSDATA_COLLECTION,
                {'_id': 'ml_resume_token'},
                {'$set': {'value': resume_token}},
                upsert=True)
        if not resume:
            # get resume_token
            resume_token = self.mongodb.getCurrentResumeToken(
                PROPERTIES_COLLECTION)
            # find all available ids
            cursor = self.mongodb.collection(
                PROPERTIES_COLLECTION).find(query, {'_id': 1})
            available_ids = [doc['_id'] for doc in cursor]
            # split results into multiple batches to save memory and speed up.
            # 1M use 20~30GB memory. 100k would be a good batch size.
            # predict and save to database
            while available_ids:
                batch_ids = available_ids[:100000]
                available_ids = available_ids[100000:]
                self.predict(batch_ids, True)
            # self.predict(available_ids, True)
            # update resume_token
            save_resume_token(resume_token)
        else:
            ml_resume_token = self.mongodb.findOne(
                SYSDATA_COLLECTION, {'_id': 'ml_resume_token'})
            if ml_resume_token is None:
                logger.error('no resume token found')
                self.__update_status('watch', 'error')
                return
            resume_token = ml_resume_token['value']
        if resume_token is None:
            raise Exception('no resume token found')
        # watch from resume_token
        with self.mongodb.watch(PROPERTIES_COLLECTION, resume_token) as stream:
            changeIDs = []
            # take care of the SIGINT

            def before_close(signum, frame):
                signame = Signals(signum).name
                logger.info(
                    f'Signal handler called with signal {signame} ({signum})')
                if len(changeIDs) > 0:
                    self.predict(changeIDs, True)
                    changeIDs.clear()
                save_resume_token(stream.resume_token)
                stream.close()
                self.__update_status('watch', 'interrupted')
                logger.info('watching interrupted')
            signal(SIGINT, before_close)
            # start watching
            sleepCount = 0
            while stream.alive:
                try:
                    change = stream.try_next()
                except StopIteration:
                    logger.warn('watching stopped')
                    # TODO: skip this record and continue
                    continue
                except Exception as e:
                    logger.error(e)
                    continue
                if change is not None:
                    # logger.info(f"change {change['operationType']}")
                    if change['operationType'] == 'insert':
                        changeIDs.append(change['documentKey']['_id'])
                    elif change['operationType'] == 'update':
                        if 'updateDescription' in change:
                            updatedFields = change['updateDescription']['updatedFields']
                            for field in WATCH_FIELDS:
                                if field in updatedFields:
                                    changeIDs.append(
                                        change['documentKey']['_id'])
                                    break
                        else:
                            logger.info(
                                f"no updateDescription {change['documentKey']['_id']}")
                    else:
                        logger.info(
                            f"no need to predict {change['documentKey']['_id']} {change['operationType']}")
                    continue
                if len(changeIDs) > 0:
                    logger.info(f"predict {len(changeIDs)} ids")
                    self.predict(changeIDs, True)
                    changeIDs = []
                else:
                    if sleepCount >= 10:
                        print(".", end='', flush=True)
                        sleepCount = 0
                    sleepCount += 1
                    time.sleep(6)
        self.__update_status('watch', 'done')


def main(debug: bool = False):
    """Main function."""
    logger.info('start organizer...')
    orgnizer = Organizer()
    orgnizer.load_data()
    orgnizer.init_transformers()
    orgnizer.train_models()
    orgnizer.save_models()
    orgnizer.watch_n_predicting()


if __name__ == '__main__':
    main(True)
