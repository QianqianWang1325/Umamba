# Umamba
Umamba

1.Data preparation
Taking the Potsdam dataset as an example Potsdam  
    >>--train    
    >>--val  
    >>--test   


# train
2.Train
Put the training/val/test data into the corresponding path in train.py.

For Super-Resolution and Colorization task

python train.py --task colorx2 --img_train_path ../../../../data/Potsdam_Original/train/label/ --img_test_path ../../../../data/Potsdam_Original/test/label/

For Super-Resolution task

python train.py --task srx2 --img_train_path ../../../../data/Potsdam_Original/train/label/ --img_test_path ../../../../data/Potsdam_Original/test/label/

For Colorization task

python train.py --task color --img_train_path ../../../../data/Potsdam_Original/train/label/ --img_test_path ../../../../data/Potsdam_Original/test/label/

  
# test
3.Test
Put the training/val/test data into the corresponding path in test.py.

The pre trained model generated during the training process follows the ./log_1/pkl/best. 

For Super-Resolution and Colorization task

python test.py --task colorx2 --best_pkl_path ./log/pkl/best/best.pkl

For Super-Resolution and Colorization task

python test.py --task srx2 --best_pkl_path ./log/pkl/best/best.pkl

For Super-Resolution and Colorization task

python test.py --task color --best_pkl_path ./log/pkl/best/best.pkl

The Pre-Train model can get here. [pre-train model](https://pan.baidu.com/s/1ghTCKOnyHOki6U0I0mX1WA?pwd=12SR)  Extract code: 12SR  
