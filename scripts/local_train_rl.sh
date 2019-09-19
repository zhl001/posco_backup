python train.py --algo sac -n 50000 -vae vae300.pkl --local-control --donkey-name=my_robot1234 --broker=localhost

python train.py --algo sac -n 50000 -vae vae1800.pkl -i DonkeyVae-v0-level-0.pkl --donkey-name=my_robot1234 --broker=localhost --local-control
