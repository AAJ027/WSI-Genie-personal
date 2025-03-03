accesslog = '-'
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'
timeout=0 #TODO: this is a bad idea
worker_class='gthread'
threads=5
workers=1