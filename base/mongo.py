import datetime
import pandas as pd
import pymongo
from pymongo import MongoClient
from .base_cfg import BaseCfg
from bson.codec_options import DatetimeConversion

logger = BaseCfg.getLogger('mongo.py')

pd.set_option('display.max_rows', BaseCfg.pd_max_rows)
pd.set_option('display.max_columns', BaseCfg.pd_max_columns)
pd.set_option('display.width', 1000)


class MongoDB():
    def __init__(self):
        cfg = BaseCfg().getMongoCfg()
        user = cfg['user']
        password = cfg['password']
        host = cfg['host']
        dbname = cfg['name']
        port = int(cfg['port'])
        """ A util for making a connection to mongo """
        if user and password:
            mongo_uri = 'mongodb://%s:%s@%s:%s/%s?authSource=admin&authMechanism=SCRAM-SHA-1' % (
                user, password, host, port, dbname)
            logger.info(mongo_uri)
            # to avoid error: bson.errors.InvalidBSON: year 20023 is out of range
            self.conn = MongoClient(
                mongo_uri, datetime_conversion=DatetimeConversion.DATETIME_AUTO)
        else:
            logger.debug('connect to DB')
            # to avoid error: bson.errors.InvalidBSON: year 20023 is out of range
            self.conn = MongoClient(
                host, port, datetime_conversion=DatetimeConversion.DATETIME_AUTO)
        self.db = self.conn[dbname]

    def close(self):
        logger.debug('close db')
        self.conn.close()

    def getDb(self):
        return self.db

    def collection(self, collname):
        return self.db[collname]

    def cursorToDf(self, cursor, noId=False):
        list1 = []
        # 转化同时处理mapping
        result_list = list(cursor)
        df = pd.DataFrame(result_list)

        # Delete the _id
        if noId:
            del df['_id']
        return df

    def read(self, collection, query={}, noId=False):
        """ Read from Mongo and Store into DataFrame """

        if isinstance(collection, str):
            collection = self.collection(collection)
        # Make a query to the specific DB and Collection
        cursor = collection.find(query)

        # Expand the cursor and construct the DataFrame
        df = pd.DataFrame(list(cursor))
        # Delete the _id
        if noId:
            del df['_id']
        return df

    def findOne(self, collection, query):
        if isinstance(collection, str):
            collection = self.collection(collection)
        return collection.find_one(query)

    def load_data(self, collection, col_list, filter={}):
        """ Read from Mongo and Store into DataFrame with filter and col_list. Used by pred package."""
        projection = {col: 1 for col in col_list}
        cursor = self.collection(collection).find(filter, projection)
        df = pd.DataFrame(list(cursor)).infer_objects()
        print("This is the first debug!!!!!!!!!!!!!!!!!!!!!!!1\n")
        print(df.head(n=3))
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        return df

    def appendMt(self, update):
        retUpdate = {'$set': {}, **update}
        # use 'aware' datetime to making sure the time is correct
        # https://howchoo.com/g/ywi5m2vkodk/working-with-datetime-objects-and-timezones-in-python#:~:text=To%20make%20a%20datetime%20object,gives%20the%20object%20timezone%20information.
        retUpdate['$set']['_mt'] = datetime.datetime.utcnow()
        return retUpdate

    def updateOne(self, collection, query, update, upsert=None):
        # save to mongo
        update = self.appendMt(update)
        # logger.debug(query, update)
        if isinstance(collection, str):
            collection = self.collection(collection)
        if upsert is None:
            upsert = False
        #print('updateOne', query, update, upsert)
        collection.update_one(query, update, upsert)

    def update(self, collection, query, update, upsert):
        # save to mongo
        update = self.appendMt(update)
        if isinstance(collection, str):
            collection = self.collection(collection)
        collection.update_many(query, update, upsert)

    def deleteMany(self, collection, query):
        if isinstance(collection, str):
            collection = self.collection(collection)
        collection.delete_many(query)

    def replaceOne(self, collection, query, update, upsert):
        # save to mongosave
        update = self.appendMt(update)
        if isinstance(collection, str):
            collection = self.collection(collection)
        collection.replace_one(query, update, upsert)

    def save(self, collection, doc):
        # save to mongo
        try:
            doc = {**doc, '_mt': datetime.datetime.utcnow()}
            if isinstance(collection, str):
                collection = self.collection(collection)
            if '_id' in doc:
                collection.replace_one({'_id': doc['_id']}, doc, upsert=True)
                return doc['_id']
            else:
                res = collection.insert_one(doc)
                return res.inserted_id
        except pymongo.errors.PyMongoError:
            logger.error('error when save %s', doc)
            logger.error(pymongo.errors.PyMongoError)

    def insertMany(self, collection, docs):
        if isinstance(collection, str):
            collection = self.collection(collection)
        return collection.insert_many(docs)

    def rename(self, collection, targetName, dropTarget=False):
        try:
            if isinstance(collection, str):
                collection = self.collection(collection)
            collection.rename(targetName, dropTarget=dropTarget)
        except pymongo.errors.PyMongoError:
            logger.error('error when rename to  %s', targetName)
            logger.error(pymongo.errors.PyMongoError)

    def drop(self, collection):
        try:
            if isinstance(collection, str):
                collection = self.collection(collection)
            collection.drop()
        except pymongo.errors.PyMongoError:
            logger.error('error when drop to  %s', collection)
            logger.error(pymongo.errors.PyMongoError)

    def collRenameAndDelete(self, workingNm, targetNm):
        self.rename(self.collection(targetNm),
                    targetNm+'_backup', dropTarget=True)
        self.rename(self.collection(workingNm), targetNm, dropTarget=True)
        self.drop(self.collection(targetNm+'_backup'))

    # fields = [
    #  ("field_to_index", 1),
    #  ("second_field_indexed", -1)
    #  ]
    def createIndex(self, collectionName, fields, unique=False):
        try:
            self.collection(collectionName).create_index(
                fields, unique=unique, background=True)
        except pymongo.errors.PyMongoError:
            logger.error('error when create index to  %s', fields)
            logger.error(pymongo.errors.PyMongoError)

    def hasIndex(self, collectionName, fields):
        indexes = self.collection(collectionName).index_information()
        logger.info('hasIndex', indexes)
        return indexes.get(fields)

    def getCurrentResumeToken(self, collectionName):
        resume_token = None
        with self.collection(collectionName).watch() as stream:
            while stream.alive:
                resume_token = stream.resume_token
                break
        return resume_token

    def watch(self, collectionName, resume_token):
        return self.collection(collectionName).watch(resume_after=resume_token)


theMongoDb = MongoDB()


def getMongoClient():
    return theMongoDb
