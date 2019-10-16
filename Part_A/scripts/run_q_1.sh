cd .. | return 1
export CUDA_VISIBLE_DEVICES=0
python main.py -baseline true >logs/train_baseline.log
unset CUDA_VISIBLE_DEVICES
