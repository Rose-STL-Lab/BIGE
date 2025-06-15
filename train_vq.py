import os
import json
# from osim_sequence import OSIMSequence,load_osim
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import models.vqvae as vqvae
import utils.losses as losses 
import options.option_vq as option_vq
import utils.utils_model as utils_model
from dataset import dataset_MOT_MCS, dataset_TM_eval, dataset_MOT_segmented
import utils.eval_trans as eval_trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')
from utils.word_vectorizer import WordVectorizer
# import nimblephysics as nimble
import deepspeed
os.environ["MASTER_ADDR"] = "127.0.0.1"  # Default is localhost
os.environ["MASTER_PORT"] = "30000" 

import logging
logging.getLogger("deepspeed").setLevel(logging.DEBUG)

from osim_sequence import load_osim, groundConstraint, GetLowestPointLayer

def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):

    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr

def get_foot_losses(motion, y_translation=0.0,feet_threshold=0.01):
    # y_translation = 0.0
    min_height, idx = motion[..., 1].min(dim=-1)

    # y_translation = -min_height.median() # Change reference to median  (Other set of experiments determine this. See paper)

    # print(min_height,idx,motion[..., 1].shape)
    min_height = min_height + y_translation
    pn = -torch.minimum(min_height, torch.zeros_like(min_height))  # penetration
    pn[pn < feet_threshold] = 0.0
    fl = torch.maximum(min_height, torch.zeros_like(min_height))  # float
    fl[fl < feet_threshold] = 0.0
    bs, t = idx.shape

    I = torch.arange(bs).view(bs, 1).expand(-1, t-1).long()
    J = torch.arange(t-1).view(1, t-1).expand(bs, -1).long()
    J_next = J + 1
    feet_motion = motion[I, J, idx[:, :-1]]
    feet_motion_next = motion[I, J_next, idx[:, :-1]]
    sk = torch.norm(feet_motion - feet_motion_next, dim=-1)
    contact = fl[:, :t] < feet_threshold

    sk = sk[contact[:, :-1]]  # skating
    # action: measure the continuity between frames
    vel = motion[:, 1:] - motion[:, :-1]
    acc = vel[:, 1:] - vel[:, :-1]
    acc = torch.norm(acc, dim=-1)
    # all losses
    loss_pn = pn[:, :t].view(-1)
    loss_fl = fl[:, :t].view(-1)
    loss_sk = sk.view(-1)
    metr_act = acc[:, :t].view(-1)

    return loss_pn.sum()/bs, loss_fl.sum()/bs, loss_sk.sum()/bs

##### ---- Exp dirs ---- #####
args = option_vq.get_args_parser()
torch.manual_seed(args.seed)
torch.cuda.set_device(args.local_rank)

# assert os.path.isdir(args.subject), "Location to subject info does not exist"
osim_path = os.path.join(args.subject,'OpenSimData','Model', 'LaiArnoldModified2017_poly_withArms_weldHand_scaled.osim')
assert os.path.exists(osim_path), f"Osim file:{osim_path} does not exist"
osim_geometry_dir = os.path.join("/data/panini/MCS_DATA",'OpenCap_LaiArnoldModified2017_Geometry')
assert os.path.exists(osim_geometry_dir), f"Osim geometry path:{osim_geometry_dir} does not exist"

osim = load_osim(osim_path, osim_geometry_dir, ignore_geometry=False)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok = True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

w_vectorizer = WordVectorizer('./glove', 'our_vab')

if args.dataname == 'kit' : 
    dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt'  
    args.nb_joints = 21
    
else :
    dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
    args.nb_joints = 22

logger.info(f'Training on {args.dataname}, motions are with {args.nb_joints} joints')

# wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
# eval_wrapper = EvaluatorModelWrapper(wrapper_opt)


##### ---- Dataloader ---- #####
# train_loader = dataset_MOT_MCS.DATALoader(args.dataname,
#                                         args.batch_size,
#                                         window_size=args.window_size,
#                                         unit_length=2**args.down_t)

train_loader = dataset_MOT_segmented.DATALoader(args.dataname,
                                        args.batch_size,
                                        window_size=args.window_size,
                                        unit_length=2**args.down_t)

# train_loader_iter = dataset_MOT_MCS.cycle(train_loader)
train_loader_iter = dataset_MOT_segmented.cycle(train_loader)

# val_loader = dataset_TM_eval.DATALoader(args.dataname, False,
#                                         32,
#                                         w_vectorizer,
#                                         unit_length=2**args.down_t)

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


# if args.resume_pth : 
#     logger.info('loading checkpoint from {}'.format(args.resume_pth))
#     ckpt = torch.load(args.resume_pth, map_location='cuda')
#     net.load_state_dict(ckpt['net'], strict=True)
if args.resume_pth : 
    logger.info('loading checkpoint from {}'.format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location='cpu')
    new_state_dict = {k.replace("module.", ""): v for k, v in ckpt['net'].items()}
    net.load_state_dict(new_state_dict, strict=True)
net.train()
net.cuda()

##### ---- Optimizer & Scheduler ---- #####
optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

deepspeed_config = {
    "train_micro_batch_size_per_gpu": args.batch_size,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": args.lr,
            "betas": [
                0.9,
                0.99
            ],
            "weight_decay":args.weight_decay
        }
    },
    "gradient_accumulation_steps": 1,
    # "fp16": {
    #     "enabled": True
    # },
    "zero_optimization": {
        "stage": 0
    }, 
    "communication_data_type": "fp16",  # Try different precision
    "communication_backend": "nccl"  # Try "mpi" if available
}
net, optimizer, _, _ = deepspeed.initialize(model=net, optimizer=optimizer, args=args, config_params=deepspeed_config)

Loss = losses.ReConsLoss(args.recons_loss, args.nb_joints, args.loss_mode)

##### ------ warm-up ------- #####
avg_recons, avg_perplexity, avg_commit, avg_temporal = 0., 0., 0., 0.
avg_trans, avg_rot, avg_penetration = 0., 0., 0.

for nb_iter in range(1, args.warm_up_iter):
    
    optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, args.warm_up_iter, args.lr)
    
    gt_motion,_, names = next(train_loader_iter)
    gt_motion = gt_motion.cuda().float() # (bs, 64, dim)

    pred_motion, loss_commit, perplexity = net(gt_motion)
    if args.loss_mode == 'together':
        loss_motion = Loss(pred_motion, gt_motion)
    elif args.loss_mode == 'separate':
        loss_motion, rot_loss, trans_loss = Loss(pred_motion, gt_motion)
    
    loss_temp = torch.mean((pred_motion[:,1:,:]-pred_motion[:,:-1,:])**2)
    
    # loss_penetration = Loss.penetration_loss(pred_motion, osim)
    
    # loss_vel = Loss.forward_vel(pred_motion, gt_motion)
    # loss_pn, loss_fl, loss_sk = get_foot_losses(pred_motion)
    # print(loss_pn, loss_fl, loss_sk)
    
    # # hip flexion 7, 15
    # hip_flexion_l = -pred_motion[:,:,7].max(dim=1).values.mean()
    # hip_flexion_r = -pred_motion[:,:,15].max(dim=1).values.mean()
    # # knee angle 10, 18
    # knee_angle_l = -pred_motion[:,:,10].max(dim=1).values.mean()
    # knee_angle_r = -pred_motion[:,:,18].max(dim=1).values.mean()
    # # ankle angle 12, 20
    # ankle_angle_l = -pred_motion[:,:,12].max(dim=1).values.mean()
    # ankle_angle_r = -pred_motion[:,:,20].max(dim=1).values.mean()
    
    # print(hip_flexion_l, hip_flexion_r, knee_angle_l, knee_angle_r, ankle_angle_l, ankle_angle_r)
    
    loss = loss_motion + args.commit * loss_commit + args.temporal * loss_temp # + args.penetration * loss_penetration #+ args.loss_vel * loss_vel 
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    avg_recons += loss_motion.item()
    avg_perplexity += perplexity.item()
    avg_commit += loss_commit.item()
    avg_temporal += loss_temp.item()
    # avg_penetration += loss_penetration.item()
    
    if args.loss_mode == 'separate':
        avg_rot += rot_loss.item()
        avg_trans += trans_loss.item()
    
    if nb_iter % args.print_iter ==  0 :
        avg_recons /= args.print_iter
        avg_perplexity /= args.print_iter
        avg_commit /= args.print_iter
        avg_temporal /= args.print_iter
        
        if args.loss_mode == 'separate':
            avg_rot += rot_loss.item()
            avg_trans += trans_loss.item()
        
        if args.loss_mode == 'together':
            logger.info(f"Warmup. Iter {nb_iter} :  lr {current_lr:.5f} \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f} \t Temporal. {avg_temporal:.5f}") # \t Penetration. {avg_penetration:.5f}")
            avg_recons, avg_perplexity, avg_commit, avg_temporal = 0., 0., 0., 0.
        
        if args.loss_mode == 'separate':
            logger.info(f"Warmup. Iter {nb_iter} :  lr {current_lr:.5f} \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f} \t Temporal. {avg_temporal:.5f} \t Rot Loss {avg_rot:.2f} \t Trans Loss {avg_trans:.2f}") # \t Penetration. {avg_penetration:.5f}")
            avg_recons, avg_perplexity, avg_commit, avg_temporal, avg_rot, avg_trans = 0., 0., 0., 0., 0., 0.              

##### ---- Training ---- #####
avg_recons, avg_perplexity, avg_commit, avg_temporal = 0., 0., 0., 0.
avg_trans, avg_rot, avg_penetration = 0., 0., 0.
torch.save({'net' : net.state_dict()}, os.path.join(args.out_dir, 'warmup.pth'))
# best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = eval_trans.evaluation_vqvae(args.out_dir, val_loader, net, logger, writer, 0, best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, eval_wrapper=eval_wrapper)

for nb_iter in range(1, args.total_iter + 1):
    
    gt_motion,_,_ = next(train_loader_iter)
    gt_motion = gt_motion.cuda().float() # bs, nb_joints, joints_dim, seq_len
    
    pred_motion, loss_commit, perplexity = net(gt_motion)
    if args.loss_mode == 'together':
        loss_motion = Loss(pred_motion, gt_motion)
    elif args.loss_mode == 'separate':
        loss_motion, rot_loss, trans_loss = Loss(pred_motion, gt_motion)
    
    loss_temp = torch.mean((pred_motion[:,1:,:]-pred_motion[:,:-1,:])**2)
    
    # loss_penetration = Loss.penetration_loss(pred_motion, osim)
    # loss_vel = Loss.forward_vel(pred_motion, gt_motion)
    # loss_pn, loss_fl, loss_sk = get_foot_losses(pred_motion)
    # print(loss_pn, loss_fl, loss_sk)
    
    loss = loss_motion + args.commit * loss_commit + args.temporal * loss_temp # + args.penetration * loss_penetration #+ args.loss_vel * loss_vel # Need to remove/change loss_vel since its not SMPL. original weight of loss temp was 0.5.
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    avg_recons += loss_motion.item()
    avg_perplexity += perplexity.item()
    avg_commit += loss_commit.item()
    avg_temporal += loss_temp.item()
    # avg_penetration += loss_penetration.item()
    
    if args.loss_mode == 'separate':
        avg_rot += rot_loss.item()
        avg_trans += trans_loss.item()
    
    if nb_iter % args.print_iter ==  0 :
        avg_recons /= args.print_iter
        avg_perplexity /= args.print_iter
        avg_commit /= args.print_iter
        avg_temporal /= args.print_iter
        avg_penetration /= args.print_iter
        
        writer.add_scalar('./Train/L1', avg_recons, nb_iter)
        writer.add_scalar('./Train/PPL', avg_perplexity, nb_iter)
        writer.add_scalar('./Train/Commit', avg_commit, nb_iter)
        
        if args.loss_mode == 'separate':
            avg_rot += rot_loss.item()
            avg_trans += trans_loss.item()
        
        if args.loss_mode == 'together':
            logger.info(f"Warmup. Iter {nb_iter} :  lr {current_lr:.5f} \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f} \t Temporal. {avg_temporal:.5f}") # \t Penetration. {avg_penetration:.5f}")
            avg_recons, avg_perplexity, avg_commit, avg_temporal, avg_penetration = 0., 0., 0., 0., 0.
        
        if args.loss_mode == 'separate':
            logger.info(f"Warmup. Iter {nb_iter} :  lr {current_lr:.5f} \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f} \t Temporal. {avg_temporal:.5f} \t Rot Loss {avg_rot:.2f} \t Trans Loss {avg_trans:.2f}") # \t Penetration. {avg_penetration:.5f}")
            avg_recons, avg_perplexity, avg_commit, avg_temporal, avg_rot, avg_trans, avg_penetration = 0., 0., 0., 0., 0., 0., 0.
    
    if nb_iter % (10*args.eval_iter) == 0:
        torch.save({'net' : net.state_dict()}, os.path.join(args.out_dir, str(nb_iter) + '.pth'))

    # if nb_iter % args.eval_iter==0 :
    #     # The line `best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching,
    #     # writer, logger = eval_trans.evaluation_vqvae(args.out_dir, val_loader, net, logger, writer,
    #     # nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching,
    #     # eval_wrapper=eval_wrapper)` is calling a function named `evaluation_vqvae` from the
    #     # `eval_trans` module.
    #     best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = eval_trans.evaluation_vqvae(args.out_dir, val_loader, net, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, eval_wrapper=eval_wrapper)
        