#!/bin/bash
#SBATCH -J train_ML
#SBATCH --mem=32G
#SBATCH --output=logs/train_ML_%j.out
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH -t 60:00:00
#SBATCH --gres=gpu:1

cd ..

python train.py
#python eval.py
