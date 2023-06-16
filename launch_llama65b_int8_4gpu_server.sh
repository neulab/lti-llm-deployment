	TOKENIZERS_PARALLELISM=false \
	MODEL_NAME=huggyllama/llama-65b \
	MODEL_CLASS=AutoModelForCausalLM \
	DEPLOYMENT_FRAMEWORK=hf_accelerate \
	DTYPE=int8 \
	MAX_INPUT_LENGTH=1024
	MAX_BATCH_SIZE=4 \
	CUDA_VISIBLE_DEVICES=0,1,2,3 \
	gunicorn -t 0 -w 1 -b 0.0.0.0:5000 inference_server.server:app --access-logfile - --access-logformat '%(h)s %(t)s "%(r)s" %(s)s %(b)s'