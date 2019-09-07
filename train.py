import Config
from Core.dataset import DatasetFactory
from Core.tasks import Segmentation
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str)
    args = parser.parse_args()

    ##############################################################################################
    cfg = getattr(Config, args.model)()
    dataset = DatasetFactory(cfg.train.csv_path, cfg)
    train_loader = dataset.yield_loader(is_test=False)
    val_loader = dataset.yield_loader(is_test=True)

    # Print params
    print('#####################################')
    print('MODE: train\n')
    print('NET: {}'.format(cfg.model.architecture.__name__))
    print('TGT_SIZE: {}'.format(cfg.image.tgt_size))
    print('LOSS: {}'.format(type(cfg.loss).__name__))
    print('BATCH_SIZE: {}'.format(cfg.image.batch_size))
    print('OPTIMIZER: {}'.format(cfg.optim.type))
    print('WEIGHT_DECAY: {}'.format(cfg.optim.weight_decay))
    print('#####################################')

    ##########################################################################################
    #################################### TRAINING ############################################
    ##########################################################################################
    net = cfg.model.architecture(pretrained=cfg.model.pretrained)

    trainer = Segmentation(net,
                           mode='train',
                           criterion=cfg.loss,
                           debug=False,
                           fold=cfg.fold)

    # if cfg.model.weights is not None:
    #     trainer.load_model(cfg.model.weights)

    # Cycle 1
    trainer.create_optmizer(optimizer=cfg.optim.type,
                            lr=cfg.train.cycle1.lr,
                            scheduler=cfg.train.cycle1.scheduler,
                            T_max=cfg.train.cycle1.tmax,
                            T_mul=cfg.train.cycle1.tmul,
                            lr_min=cfg.train.cycle1.lr_min,
                            freeze_encoder=cfg.train.cycle1.freeze_encoder,
                            freeze_bn=cfg.train.cycle1.freeze_bn)
    trainer.train_network(train_loader, val_loader,
                          grad_acc=cfg.train.cycle1.grad_acc,
                          n_epoch=cfg.train.cycle1.n_epoch,
                          epoch_per_val=cfg.train.epoch_per_val,
                          mixup=cfg.image.mixup)

    # Cycle 2
    trainer.create_optmizer(optimizer=cfg.optim.type,
                            lr=cfg.train.cycle2.lr,
                            scheduler=cfg.train.cycle2.scheduler,
                            T_max=cfg.train.cycle2.tmax,
                            T_mul=cfg.train.cycle2.tmul,
                            lr_min=cfg.train.cycle2.lr_min,
                            freeze_encoder=cfg.train.cycle2.freeze_encoder,
                            freeze_bn=cfg.train.cycle2.freeze_bn)

    trainer.train_network(train_loader, val_loader,
                          grad_acc=cfg.train.cycle2.grad_acc,
                          n_epoch=cfg.train.cycle2.n_epoch,
                          epoch_per_val=cfg.train.epoch_per_val,
                          mixup=cfg.image.mixup)
