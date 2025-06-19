import os
import json

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import models.vqvae as vqvae
import options.option_vq as option_vq
import utils.utils_model as utils_model
from dataset import dataset_TM_eval, dataset_MOT_MCS, dataset_MOT_segmented
import utils.eval_trans as eval_trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')
import numpy as np
##### ---- Exp dirs ---- #####
args = option_vq.get_args_parser()
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok = True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))


from utils.word_vectorizer import WordVectorizer
w_vectorizer = WordVectorizer('./glove', 'our_vab')


dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if args.dataname == 'kit' else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'

wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
# eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
eval_wrapper = None


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##### ---- Dataloader ---- #####
args.nb_joints = 21 if args.dataname == 'kit' else 22
args.nb_joints = 23 # fixed issues

##### ---- Network ---- #####
net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                       args.nb_code,
                       args.code_dim,
                       args.output_emb_width,
                       args.down_t,
                       args.stride_t,
                       args.width,
                       args.depth,
                       args.dilation_growth_rate,
                       args.vq_act,
                       args.vq_norm)

assert args.resume_pth is not None, "Cannot run the optimization without a trained VQ-VAE"
logger.info('loading checkpoint from {}'.format(args.resume_pth))
ckpt = torch.load(args.resume_pth, map_location='cpu')
new_state_dict = {k.replace("module.", ""): v for k, v in ckpt['net'].items()}
net.load_state_dict(new_state_dict, strict=True)
net.eval()
net.to(device)

# val_loader = dataset_TM_eval.DATALoader(args.dataname, True, 32, w_vectorizer, unit_length=2**args.down_t)
# if args.dataname == 'mcs':
# val_loader = dataset_MOT_MCS.DATALoader(args.dataname,
#                                         args.batch_size,
#                                         window_size=args.window_size,
#                                         unit_length=2**args.down_t)

val_loader = dataset_MOT_segmented.addb_data_loader(
    window_size=args.window_size,
    unit_length=2**args.down_t,
    batch_size=args.batch_size,
    mode=args.dataname
)

fid = []
div = []
top1 = []
top2 = []
top3 = []
matching = []
repeat_time = 1
for i in range(repeat_time):
    eval_trans.evaluation_vqvae(args.out_dir, val_loader, net, logger, writer, 0, best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, eval_wrapper=eval_wrapper, draw=True, save=True, savenpy=(i==0))
    
    
datapath = '/home/mnt/data/addb_dataset_publication/train/No_Arm'
for dataset in os.listdir(datapath):
    for subject in os.listdir(os.path.join(datapath, dataset)):
        files = os.listdir(os.path.join(datapath, dataset, subject))
        files = [file for file in files if file.endswith('.npy') or '::trial' in file]
        if len(files) == 0: continue
        for file in files:
            os.system(f"mv {os.path.join(datapath,dataset,subject,file)} {dataset}-{subject}-{file}")




    