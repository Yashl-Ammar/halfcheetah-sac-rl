@echo off
echo Starting hyperparameter sensitivity experiments...

python train_sac.py --timesteps 500000 --lr 1e-4   --name lr_low    --seed 42
python train_sac.py --timesteps 500000 --lr 1e-3   --name lr_high   --seed 42
python train_sac.py --timesteps 500000 --gamma 0.95  --name gamma_low  --seed 42
python train_sac.py --timesteps 500000 --gamma 0.999 --name gamma_high --seed 42
python train_sac.py --timesteps 500000 --batch_size 128 --name batch_small --seed 42
python train_sac.py --timesteps 500000 --batch_size 512 --name batch_large --seed 42

echo All experiments complete.