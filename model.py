import os
from os import path, makedirs
import torch

from simclr import SimCLR
from simclr.modules import LARS


def load_optimizer(args, model):

    scheduler = None
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # TODO: LARS
    elif args.optimizer == "LARS":
        # optimized using LARS with linear learning rate scaling
        # (i.e. LearningRate = 0.3 × BatchSize/256) and weight decay of 10−6.
        learning_rate = 0.3 * args.batch_size / 256
        optimizer = LARS(
            model.parameters(),
            lr=learning_rate,
            weight_decay=args.weight_decay,
            exclude_from_weight_decay=["batch_normalization", "bias"],
        )

        # "decay the learning rate with the cosine decay schedule without restarts"
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs, eta_min=0, last_epoch=-1
        )
    else:
        raise NotImplementedError

    return optimizer, scheduler


def save_model(args, model, optimizer):
    #project_root = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
    project_root = path.dirname(path.abspath(__file__))
    base_dir = path.join(
        f'{args.dataset}',
        f'{args.optimizer}',
        f'{args.lr}',
        f'{args.temperature}',
        f'Conditional_{args.conditional_contrastive}'
    )
    models_dir = path.join(project_root, 'save', base_dir)
    makedirs(models_dir, exist_ok=True)
    out = os.path.join(models_dir, "checkpoint_{}.tar".format(args.current_epoch))

    # To save a DataParallel model generically, save the model.module.state_dict().
    # This way, you have the flexibility to load the model any way you want to any device you want.
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), out)
    else:
        torch.save(model.state_dict(), out)

# Path of model to load when fine-tuning
def load_model_path(args):
    project_root = path.dirname(path.abspath(__file__))
    base_dir = path.join(
        f'{args.dataset}',
        f'{args.optimizer}',
        f'{args.lr}',
        f'{args.temperature}',
        f'Conditional_{args.conditional_contrastive}'
    )
    models_dir = path.join(project_root, 'save', base_dir)
    return os.path.join(models_dir, "checkpoint_{}.tar".format(args.epoch_num))
