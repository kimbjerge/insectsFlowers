# conda activate FairMOT
# batch-size 16 on Balser uses 37 GByte GPU memory
python main.py --data-directory ../datasets/Flowers --exp_directory FExp --epochs 30 --batch-size 16
