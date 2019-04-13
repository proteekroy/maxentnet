import sys
sys.path.append('../')
from torch.utils.data import Dataset
from data_loader.dataloader import *
import torch.utils.data
from trainer.maxent_arl_trainer import MaxentNet
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "6"
use_cuda = torch.cuda.is_available()
# device = torch.device("cuda") if use_cuda else "cpu"
# print('==> Loading and Preparing data..')
# data = AdultDataLoader()
# data = ExtendedYaleBDataLoader()
# data = GermanDataLoader()

# alpha = 0.00001  # 1 - (1/data.n_target_class)
# alphalist = np.linspace(0.32, 0.38, num=4)  # np.asarray([alpha])  # np.linspace(0, width, num=30)
# alphalist = np.flip(alphalist)
# alphalist = [0]
runs = np.arange(5, 6)
# namelist = ['ml_cifar100_0.30','ml_cifar100_0.4','ml_cifar100_0.5','ml_cifar100_0.60']
# namelist = ['cifar1000.2','cifar1000.30','cifar1000.4','cifar1000.5','cifar1000.60']
# namelist = ['cifar10_0.60']
# namelist = ['ml_cifar10_0.1','ml_cifar10_0.2','ml_cifar10_0.3','ml_cifar10_0.4','ml_cifar10_0.5','ml_cifar10_0.60']
# namelist = ['cifar10_0.1','cifar10_0.2','cifar10_0.3','cifar10_0.4','cifar10_0.5','cifar10_0.60']


# namelist = ['ml_cifar10_0.1','ml_cifar10_0.2','ml_cifar10_0.30','ml_cifar10_0.4','ml_cifar10_0.5','ml_cifar10_0.60']
# namelist = ['cifar10_0.1','cifar10_0.2','cifar10_0.30','cifar10_0.4','cifar10_0.5','cifar10_0.60']


# Runs 2 to 5 needs to be done for these
# namelist = ['ml_cifar100_0.0']
# namelist = ['cifar1000.1']
# for r in runs:
embed_length_list = [128] # [256, 64, 2]
for embed_length in embed_length_list:
#for p_options in range(1, 3):
        alpha = 0.3
        data = CIFAR100DataLoader(embed_length=embed_length)
        # data = MNISTDataLoader()
        # data = CIFAR100DataLoader()
        # data = GaussianDataLoader()
        # data = Gaussian3DDataLoader()
        print('==> Building models..')
        data.load()  # <--This method loads all the parameters for the data including neural-net models and optimizer

        trainloader = torch.utils.data.DataLoader(data.trainset, batch_size=data.train_batch_size, shuffle=True,
                                                  num_workers=40)
        testloader = torch.utils.data.DataLoader(data.testset, batch_size=data.test_batch_size, shuffle=False,
                                                 num_workers=40)

    # for alpha in alphalist:
    # for name_iter in namelist:
        # name = 'cifar100_' + str(float(alpha))[:4]+'_'+str(r)
        name = 'np_cifar100_z_'+str(embed_length)
        # if p_options == 1:
        #     name = 'cifar10_z_2'
        # else:
        #     name = 'ml_cifar10_z_2'
        # name = name_iter+'_'+str(r)
        target_name = name + "_target_" + ".ckpt"
        exist = os.path.isfile('../checkpoint/'+target_name)
        # name = 'german' + str(float(alpha))[:4]
        if exist:
            continue
        # print(name)
        trainer = MaxentNet(
                    data,
                    train_loader=trainloader,
                    test_loader=testloader,
                    total_epoch=150,
                    alpha=alpha,
                    use_cuda=use_cuda,
                    ckpt_filename=name,
                    privacy_flag=False,
                    privacy_option=1,
                    resume=False,
                    resume_filename=name+'.ckpt',
                    print_interval_train=10,
                    print_interval_test=10
        )
        #
        trainer.train()
        # trainer.train_without_sensitive_label()
        trainer.train_adversary(model_filename=name+'.ckpt', total_epoch=150)
        trainer.train_target(model_filename=name + '.ckpt', total_epoch=100)