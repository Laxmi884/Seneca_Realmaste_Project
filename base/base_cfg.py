
# TODO: consistent file naming in camelCase or snake_case naming
from datetime import datetime
from lib2to3.pytree import Base
import os
import importlib.util
import sys
import logging


""" get mongo config environment variables """


class BaseCfg():
    pd_max_rows = 20
    pd_max_columns = 20
    debug = False
    rootLogger = None

    def __init__(self):
        self.module = None
        self.debug = BaseCfg.isDebug()

    @staticmethod
    def isDebug():
        BaseCfg.debug = ('debug' in sys.argv) or os.environ.get('DEBUG')
        return BaseCfg.debug

    @staticmethod
    def getLogger(name):
        if BaseCfg.rootLogger is None:
            logging.basicConfig()  # level=logging.NOTSET)
            logging.root.setLevel(logging.NOTSET)
            BaseCfg.rootLogger = logging.root  # getLogger()
            if BaseCfg.isDebug():
                BaseCfg.rootLogger.setLevel(logging.DEBUG)
            BaseCfg.rootLogger.debug("root logger created")
        if name is None:
            return BaseCfg.rootLogger
        logger = logging.getLogger(name)
        if BaseCfg.isDebug():
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        return logger

    @staticmethod
    def getTs():
        return datetime.now()

    def getConfig(self):
        """ get the configuration module """
        # TODO: shall save loaded module for reusing, unless has reload flag.
        base_path = os.environ['RMBASE_FILE_PYTHON']
        module_name = 'base'
        spec = importlib.util.spec_from_file_location(module_name, base_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # cfg = module.mongoConfig['theone']
        return module

    def getMongoCfg(self):
        module = self.getConfig()
        return module.mongoConfig['theone']

    def getModelPath(self):
        # TODO: shall use getConfig() to get the module
        base_path = os.environ['RMBASE_FILE_PYTHON']
        module_name = 'base'
        spec = importlib.util.spec_from_file_location(module_name, base_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modelPath = module.modelPath
        return modelPath

    def getLogConfig(self):
        module = self.getConfig()
        return module.logConfig


if __name__ == '__main__':
    mm = BaseCfg().getModelPath()
