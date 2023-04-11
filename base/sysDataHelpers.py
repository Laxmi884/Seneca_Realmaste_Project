from logging import getLogger
import math as Math
import datetime
import sys
import socket
import os
import signal
COMPLETE_STRING = 'complete'

logger = getLogger('sysDataHelper')

# args:
# Sysdata, collection, required
# id required
# raise:
# expection when no id


def killCurrentProcess(db, Sysdata, id):
    if not id:
        raise ValueError('Id required')
    ret = Sysdata.find_one({'_id': id})
    if ret is None:
        return
    else:
        retStatus = ret.get('status')
        retPid = ret.get('pid')
        curPid = os.getpid()
        if (retStatus == 'running') and (retPid != curPid):
            try:
                os.kill(retPid, signal.SIGKILL)
                logger.info("kill runnning process %d", retPid)
                setSysdataTs(db, Sysdata, {
                    'endTs': datetime.datetime.now(),
                    'status': COMPLETE_STRING
                })
            except Exception as e:
                logger.info('error when kill running process')
                logger.info(e)


# args:
# Sysdata, collection, required
# params { id:required, batchNm, 'endTs','startTs','status'}
# raise:
# expection when no id

def setSysdataTs(db, Sysdata, params={}):
    id = params.get('id')
    if id is None:
        raise ValueError('Id Required')
    obj = {
        'hostName': socket.gethostname(),
        'pid': os.getpid()
    }
    for fld in ['batchNm', 'endTs', 'startTs', 'status']:
        val = params.get(fld)
        if val:
            obj[fld] = val
    query = {'_id': id}
    update = {'$set': obj}
    db.updateOne(Sysdata, query, update, True)


if __name__ == '__main__':

    pass
