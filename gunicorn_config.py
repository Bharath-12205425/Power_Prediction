import multiprocessing
import os

bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"

workers = 1

worker_class = 'sync'

threads = 2

timeout = 120
graceful_timeout = 30
keepalive = 2

max_requests = 100
max_requests_jitter = 20

accesslog = '-'
errorlog = '-'
loglevel = 'info'

proc_name = 'power_prediction'

preload_app = True

limit_request_line = 4096
limit_request_fields = 100
limit_request_field_size = 8190
