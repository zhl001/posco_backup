python train.py --algo sac -n 50000 -vae ~/Desktop/vae1800.pkl -i logs/sac/DonkeyVae-v0-level-3_13/DonkeyVae-v0-level-3_best.pkl --teleop --donkey-name=my_robot1234 --broker=192.168.0.24 --local-control

python train.py --algo sac -n 50000 -vae ~/Desktop/vae1800.pkl -i ~/Desktop/DonkeyVae-v0-level-0.pkl --teleop --donkey-name=my_robot1234 --broker=192.168.0.24 --local-control

python train.py --algo sac -n 50000 -vae ~/Desktop/vae300.pkl -i logs/sac/DonkeyVae-v0-level-0_2/DonkeyVae-v0-level-0.pkl --teleop --donkey-name=my_robot1234 --broker=192.168.0.24 --local-control

python train.py --algo sac -n 50000 -vae ~/Desktop/190915_vae.pkl -i ~/Desktop/aaalmost_sac.pkl --teleop --donkey-name=my_robot1234 --broker=192.168.0.24 --local-control
