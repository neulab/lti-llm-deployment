	TOKENIZERS_PARALLELISM=false \
	MODEL_NAME=chavinlo/alpaca-native \
	MODEL_CLASS=AutoModelForCausalLM \
	DEPLOYMENT_FRAMEWORK=hf_accelerate \
	DTYPE=fp16 \
	MAX_INPUT_LENGTH=1024 \
	MAX_BATCH_SIZE=3 \
	CUDA_VISIBLE_DEVICES=0,1 \
	gunicorn -t 0 -w 1 -b 0.0.0.0:5000 inference_server.server:app --access-logfile - --access-logformat '%(h)s %(t)s "%(r)s" %(s)s %(b)s'