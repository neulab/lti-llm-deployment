export MODEL_NAME=microsoft/bloom-deepspeed-inference-int8
export DEPLOYMENT_FRAMEWORK=ds_inference
export DTYPE=int8
export MAX_INPUT_LENGTH=2048
export MII_CACHE_PATH=/tmp/pfernand/mii_cache

# for more information on gunicorn see https://docs.gunicorn.org/en/stable/settings.html
gunicorn -t 0 -w 1 -b 0.0.0.0:5000 server:app --access-logfile - --access-logformat '%(h)s %(t)s "%(r)s" %(s)s %(b)s'
