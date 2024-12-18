import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for Amass',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    ## dataloader
    
    parser.add_argument('--dataname', type=str, default='mcs', help='dataset directory')
    parser.add_argument('--batch-size', default=32, type=int, help='batch size')
    parser.add_argument('--min-samples', default=10, type=int, help='number of latents to save')
    parser.add_argument('--fps', default=[20], nargs="+", type=int, help='frames per second')
    parser.add_argument('--seq-len', type=int, default=128, help='training motion length')
    parser.add_argument('--window-size', type=int, default=64)
    parser.add_argument('--num-runs', default = 5, type=int, help='number of runs')
    
    ## optimization 
    parser.add_argument('--total-iter', default=50000, type=int, help='number of total iterations to run')
    parser.add_argument('--warm-up-iter', default=1000, type=int, help='number of total iterations for warmup')
    parser.add_argument('--lr', default=1e-2, type=float, help='max learning rate')
    parser.add_argument('--lr-scheduler', default=[60000], nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=0.05, type=float, help="learning rate decay")
    
    parser.add_argument('--weight-decay', default=1e-6, type=float, help='weight decay') 
    parser.add_argument('--decay-option',default='all', type=str, choices=['all', 'noVQ'], help='disable weight decay on codebook')
    parser.add_argument('--optimizer',default='adamw', type=str, choices=['adam', 'adamw'], help='disable weight decay on codebook')
    
    ## vqvae arch
    parser.add_argument("--code-dim", type=int, default=512, help="embedding dimension")
    parser.add_argument("--nb-code", type=int, default=512, help="nb of embedding")
    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add_argument("--down-t", type=int, default=2, help="downsampling rate")
    parser.add_argument("--stride-t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=512, help="width of the network")
    parser.add_argument("--depth", type=int, default=3, help="depth of the network")
    parser.add_argument("--dilation-growth-rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument("--output-emb-width", type=int, default=512, help="output embedding width")
    parser.add_argument('--vq-act', type=str, default='relu', choices = ['relu', 'silu', 'gelu'], help='dataset directory')
    parser.add_argument('--vq-norm', type=str, default=None, help='dataset directory')
    
    ## gpt arch
    parser.add_argument("--block-size", type=int, default=25, help="seq len")
    parser.add_argument("--embed-dim-gpt", type=int, default=512, help="embedding dimension")
    parser.add_argument("--clip-dim", type=int, default=512, help="latent dimension in the clip feature")
    parser.add_argument("--num-layers", type=int, default=2, help="nb of transformer layers")
    parser.add_argument("--n-head-gpt", type=int, default=8, help="nb of heads")
    parser.add_argument("--ff-rate", type=int, default=4, help="feedforward size")
    parser.add_argument("--drop-out-rate", type=float, default=0.1, help="dropout ratio in the pos encoding")
    
    ## quantizer
    parser.add_argument("--quantizer", type=str, default='ema_reset', choices = ['ema', 'orig', 'ema_reset', 'reset'], help="eps for optimal transport")
    parser.add_argument('--quantbeta', type=float, default=1.0, help='dataset directory')

    ## other
    parser.add_argument('--print-iter', default=200, type=int, help='print frequency')
    parser.add_argument('--eval-iter', default=5000, type=int, help='evaluation frequency')
    parser.add_argument('--seed', default=69, type=int, help='seed for initializing training. ')
    parser.add_argument("--if-maxtest", action='store_true', help="test in max")
    parser.add_argument('--pkeep', type=float, default=1.0, help='keep rate for gpt training')

    ## resume
    parser.add_argument("--resume-trans", type=str, default=None, help='resume gpt pth')
    parser.add_argument("--resume-pth", type=str, default="output/VQVAE/net_best_fid.pth", help='resume vq pth')
    
    ## output directory 
    parser.add_argument('--out-dir', type=str, default='/data/panini/MCS_DATA/LIMO/', help='output directory')
    parser.add_argument('--exp-name', type=str, required=True, help='name of the experiment, will create a file inside out-dir')
    parser.add_argument('--vq-name', type=str, default="output/VQVAE/net_best_fid.pth", help='path to the vq-vae model to optimize')
    
    parser.add_argument('--data_root', default='/home/ubuntu/data/HumanML3D', type=str, help='Directory where the training and evaluation data strored in HumanML3D format')
    parser.add_argument('--topk', type=int, default=100, help='Choose the best topk results from the total number of samples')
    
    ## Loss optimization hyperparameters
    parser.add_argument('--loss-temporal', type=float, default=10, help='Hyperparameter for temporal regulaizer')
    parser.add_argument('--loss-proximity', type=float, default=0.1, help='Hyperparameter for temporal regulaizer')
    parser.add_argument('--loss-foot', type=float, default=1.0, help='Hyperparameter for temporal regulaizer')
    parser.add_argument('--feet-threshold', type=float, default=0.01, help='Hyperparameter to calculate ground whether joint in contact with the ground')
    parser.add_argument('--mcs', type = int, default = None, help='specify mcs score')
    
    parser.add_argument('--subject', type = str, default = "/data/panini/MCS_DATA/Data/000cffd9-e154-4ce5-a075-1b4e1fd66201/", help='Subject Info in opencap format')


    ############# Surrogate Model Options ################
    parser.add_argument('--low', type=float, default=0.35,   help='Muscle activation lower bound')
    parser.add_argument('--high', type=float, default=0.45,   help='Muscle activation higher bound')

    return parser.parse_args()
