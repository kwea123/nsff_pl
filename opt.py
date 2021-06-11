import argparse

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory of dataset')
    parser.add_argument('--cache_dir', type=str, default='',
                        help='cache directory')
    parser.add_argument('--dataset_name', type=str, default='monocular',
                        choices=['monocular'],
                        help='which dataset to train/val')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[512, 288],
                        help='resolution (img_w, img_h) of the image')
    parser.add_argument('--start_end', nargs='+', type=int, default=[0, 100],
                        help='start and end frames')
    parser.add_argument('--batch_from_same_image', default=False, action="store_true",
                        help='Form batch from rays of the same image. Used with monodepth supervision')

    # original NeRF parameters
    parser.add_argument('--use_viewdir', default=False, action="store_true",
                        help='whether to use view dependency in static network')
    parser.add_argument('--N_samples', type=int, default=128,
                        help='number of coarse samples')
    parser.add_argument('--N_importance', type=int, default=0,
                        help='number of additional fine samples')
    parser.add_argument('--N_emb_xyz', type=int, default=10,
                        help='number of features in xyz embedding')
    parser.add_argument('--S_emb_xyz', type=float, default=9,
                        help='max frequency in xyz embedding')
    parser.add_argument('--N_emb_dir', type=int, default=4,
                        help='number of features in dir embedding')
    parser.add_argument('--S_emb_dir', type=float, default=3,
                        help='max frequency in dir embedding')
    parser.add_argument('--perturb', type=float, default=1.0,
                        help='factor to perturb depth sampling points')
    parser.add_argument('--noise_std', type=float, default=1.0,
                        help='std dev of noise added to regularize sigma')

    # NeRF-W parameters
    parser.add_argument('--N_vocab', type=int, default=100,
                        help='''number of vocabulary (number of images) 
                                in the dataset for nn.Embedding''')
    parser.add_argument('--encode_a', default=False, action="store_true",
                        help='whether to encode appearance (NeRF-A)')
    parser.add_argument('--N_a', type=int, default=48,
                        help='number of embeddings for appearance')
    parser.add_argument('--encode_t', default=False, action="store_true",
                        help='whether to encode transient object (NeRF-U)')
    parser.add_argument('--N_tau', type=int, default=48,
                        help='number of embeddings for transient objects')
    parser.add_argument('--lambda_geo_init', type=float, default=0.04,
                        help='2d-3d flow consistency loss coefficient')
    parser.add_argument('--flow_scale', type=float, default=0.2,
                        help='flow scale to multiply to flow network output')

    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch size')
    parser.add_argument('--chunk', type=int, default=32*1024,
                        help='chunk size to split the input to avoid OOM')
    parser.add_argument('--num_epochs', type=int, default=16,
                        help='number of training epochs')

    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--num_nodes', type=int, default=1,
                        help='number of nodes')

    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint to load (including optimizers, etc)')
    parser.add_argument('--prefixes_to_ignore', nargs='+', type=str, default=['loss'],
                        help='the prefixes to ignore in the checkpoint state dict')
    parser.add_argument('--weight_path', type=str, default=None,
                        help='pretrained weight to load (do not load optimizers, etc)')

    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer type',
                        choices=['sgd', 'adam', 'radam', 'ranger'])
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='learning rate momentum')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--lr_scheduler', type=str, default='steplr',
                        help='scheduler type',
                        choices=['const', 'steplr', 'cosine', 'poly'])
    #### params for warmup, only applied when optimizer == 'sgd' or 'adam'
    parser.add_argument('--warmup_multiplier', type=float, default=1.0,
                        help='lr is multiplied by this factor after --warmup_epochs')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Gradually warm-up(increasing) learning rate in optimizer')
    ###########################
    #### params for steplr ####
    parser.add_argument('--decay_step', nargs='+', type=int, default=[20],
                        help='scheduler decay step')
    parser.add_argument('--decay_gamma', type=float, default=0.1,
                        help='learning rate decay amount')
    ###########################
    #### params for poly ######
    parser.add_argument('--poly_exp', type=float, default=0.9,
                        help='exponent for polynomial learning rate decay')
    ###########################

    # pytorch lightning parameters
    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')
    parser.add_argument('--refresh_every', type=int, default=1,
                        help='How often to refresh the progress bar')

    return parser.parse_args()
