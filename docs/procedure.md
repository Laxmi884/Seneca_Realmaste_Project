# Procedure of the program

## Worker process

- Get current offset of the topic in queue (no group)
- read db from 1 year ago to now, built datasource
- calculate baseline and delta
- save baseline and delta to db
- start listening to the topic for each **_req_** message:
  - calculate neighbors
  - evaluate property and send response to the **_res_** topic(with version number)
  - update datasource
  - send master process a **_started_** message if it is the first time

## Master process

- start worker process
- loop:
  - sleep 1 day
  - start worker process and give it a version number of YYYYMMDD
  - when received **_started_** message from the new worker process, kill old worker process
