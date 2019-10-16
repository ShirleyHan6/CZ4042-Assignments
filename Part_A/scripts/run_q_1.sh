cd .. | return 1
export CUDA_VISIBLE_DEVICES=0
python main.py -baseline true
unset CUDA_VISIBLE_DEVICES
