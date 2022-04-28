ECE740 Project

Hao Xuan, Yinsheng He, Chen Kong

In this project, we propose a new model aims for robustness.
It consists of a VAE and a classifier that work together to produce
model output. Test-time update method is also used to increase robustness
accuracy and natural accuracy.

To train model on CIFAR-10 dataset, please run
        'python train_cifar.py'
It will save last few checkpoints in './model-checkpoint' folder.

To test model on CIFAR-10 dataset, please run
        'python train_cifar.py --test-num 120'
"120" here can be replaced by any epoch number that needs to be tested

To change the test time update hyperparameters, please change the last two
arguments in the 'test()' function
        'test(vae_model, c_model, channel=channel[2], lr=0.01, num=10)'
Here, 'lr' represents learning rate and 'num' is number of steps

Training and testing on MNIST dataset is similar to CIFAR-10 dataset
train: 
        'python train_mnist_update.py'
test: 
        'python train_mnist_update.py --test-num 60'
