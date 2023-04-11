import pickle
from re import A
from base.model_store import ModelStore
from base.mongo import MongoDB
import gridfs
mongoDb = MongoDB().getDb()


def getGridFSBucket(collection):
    global mongoDb
    return gridfs.GridFSBucket(mongoDb, collection)


def getGridFS(collection):
    global mongoDb
    return gridfs.GridFS(mongoDb, collection)


class GridfsStore(ModelStore):
    """Store file in a Gridfs."""

    def __init__(
        self,
        # db: pymongo.database.Database,
        collection: str = 'fs_ml',
        prefix: str = 'ml_'
    ):
        #self.db = db
        self.collection = collection
        # self.fsbucket = gridfs.GridFSBucket(self.db, collection)
        # self.fs = gridfs.GridFS(self.db, collection)
        self.prefix = prefix

    def save_model(self, filename, data, accuracy=0, meta: dict = None):
        """Save the model."""
        filename = self.prefix + filename
        # remove old version of the file first
        bucket = getGridFSBucket(self.collection)
        for oldFile in bucket.find({"filename": filename},
                                   no_cursor_timeout=True):
            bucket.delete(oldFile._id)
        # save the new version
        if meta is None:
            meta = {
                'accuracy': accuracy,
            }
        else:
            meta['accuracy'] = accuracy
        f = bucket.open_upload_stream(
            filename=filename, metadata=meta)
        pickle.dump(data, f)
        f.close()
        return filename

    def load_model(self, filename):
        """Load the model."""
        filename = self.prefix + filename
        f = getGridFS(self.collection).get_last_version(filename=filename)
        if f is None:
            raise FileNotFoundError('No model file found.')
        meta = {}
        if f.metadata is None:
            accuracy = 0
        else:  # f.metadata is not None
            accuracy = f.metadata['accuracy'] or 0
            meta = f.metadata
        data = pickle.load(f)
        f.close()
        return (data, accuracy, meta)

    def delete_unused(self):
        """Delete unused model files."""
        pass
