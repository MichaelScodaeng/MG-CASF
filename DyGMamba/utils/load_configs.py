import argparse
import sys
import torch


def get_link_prediction_args(is_evaluation: bool = False):
    """
    get the args for the link prediction task
    :param is_evaluation: boolean, whether in the evaluation process
    :return:
    """
    # arguments
    parser = argparse.ArgumentParser('Interface for the link prediction task')
    parser.add_argument('--dataset_name', type=str, help='dataset to be used', default='wikipedia',
                        choices=['wikipedia', 'reddit', 'mooc', 'lastfm', 'enron', 'SocialEvo', 'uci'])
    parser.add_argument('--batch_size', type=int, default=200, help='batch size')
    parser.add_argument('--model_name', type=str, default='DyGMamba', help='name of the model',
                        choices=[
                            'JODIE', 'DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer', 'DyGMamba',
                            # Integrated variants for theoretical MPGNN compliance
                            'IntegratedTGAT', 'IntegratedTGN', 'IntegratedDyRep', 'IntegratedJODIE',
                            'IntegratedCAWN', 'IntegratedTCL', 'IntegratedGraphMixer', 'IntegratedDyGFormer', 'IntegratedDyGMamba'
                        ])
    parser.add_argument('--gpu', type=int, default=0, help='number of gpu to use')
    parser.add_argument('--num_neighbors', type=int, default=20, help='number of neighbors to sample for each node')
    parser.add_argument('--sample_neighbor_strategy', type=str, default='uniform', choices=['uniform', 'recent', 'time_interval_aware'], help='how to sample historical neighbors')
    parser.add_argument('--time_scaling_factor', default=1e-6, type=float, help='the hyperparameter that controls the sampling preference with time interval, '
                        'a large time_scaling_factor tends to sample more on recent links, 0.0 corresponds to uniform sampling, '
                        'it works when sample_neighbor_strategy == time_interval_aware')
    parser.add_argument('--num_walk_heads', type=int, default=8, help='number of heads used for the attention in walk encoder')
    parser.add_argument('--num_heads', type=int, default=2, help='number of heads used in attention layer')
    parser.add_argument('--num_layers', type=int, default=2, help='number of model layers')
    parser.add_argument('--walk_length', type=int, default=1, help='length of each random walk')
    parser.add_argument('--time_gap', type=int, default=2000, help='time gap for neighbors to compute node features')
    parser.add_argument('--time_feat_dim', type=int, default=100, help='dimension of the time embedding')
    parser.add_argument('--position_feat_dim', type=int, default=172, help='dimension of the position embedding')
    parser.add_argument('--time_window_mode', type=str, default='fixed_proportion', help='how to select the time window size for time window memory',
                        choices=['fixed_proportion', 'repeat_interval'])
    parser.add_argument('--patch_size', type=int, default=1, help='patch size')
    parser.add_argument('--channel_embedding_dim', type=int, default=50, help='dimension of each channel embedding')
    parser.add_argument('--max_input_sequence_length', type=int, default=32, help='maximal length of the input sequence of each node')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam', 'RMSprop'], help='name of optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--patience', type=int, default=20, help='patience for early stopping')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='ratio of validation set')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='ratio of test set')
    parser.add_argument('--num_runs', type=int, default=1, help='number of runs')
    parser.add_argument('--test_interval_epochs', type=int, default=1, help='how many epochs to perform testing once')
    parser.add_argument('--negative_sample_strategy', type=str, default='random', choices=['random', 'historical', 'inductive'],
                        help='strategy for the negative edge sampling')
    parser.add_argument('--max_interaction_times', type=int, default=10,
                        help='max interactions for src and dst to consider')
    parser.add_argument('--load_best_configs', action='store_true', default=False, help='whether to load the best configurations')
    
    # Integrated MPGNN arguments
    parser.add_argument('--fusion_strategy', type=str, default='use', 
                        choices=['use', 'caga', 'clifford', 'full_clifford', 'weighted', 'concat_mlp', 'cross_attention', 'baseline_original'],
                        help='fusion strategy for enhanced features')
    parser.add_argument('--spatial_dim', type=int, default=64, help='dimension of spatial features')
    parser.add_argument('--temporal_dim', type=int, default=64, help='dimension of temporal features')
    parser.add_argument('--ccasf_output_dim', type=int, default=128, help='output dimension of CCASF fusion')
    parser.add_argument('--use_integrated_mpgnn', action='store_true', default=True, 
                        help='whether to use integrated MPGNN approach (theoretical default)')
    parser.add_argument('--use_sequential_fallback', action='store_true', default=False,
                        help='force use of sequential (non-theoretical) approach instead of integrated MPGNN')
    
    # Embedding mode configuration
    parser.add_argument('--embedding_mode', type=str, default='none',
                        choices=['none', 'spatial_only', 'temporal_only', 'spatiotemporal_only', 'spatial_temporal', 'all'],
                        help='embedding mode for enhanced features')
    parser.add_argument('--enable_base_embedding', action='store_true', default=False,
                        help='whether to enable base learnable embedding')
    
    # Enhanced feature generation parameters
    parser.add_argument('--rpearl_hidden', type=int, default=64, help='hidden dimension for R-PEARL spatial generator')
    parser.add_argument('--rpearl_mlp_layers', type=int, default=2, help='number of MLP layers in R-PEARL')
    parser.add_argument('--rpearl_k', type=int, default=16, help='K parameter for R-PEARL')
    parser.add_argument('--lete_hidden', type=int, default=64, help='hidden dimension for LeTE temporal generator')
    parser.add_argument('--lete_layers', type=int, default=2, help='number of layers in LeTE')
    parser.add_argument('--lete_p', type=float, default=0.5, help='dropout probability for LeTE')
    
    # Fusion-specific parameters
    parser.add_argument('--use_hidden_dim', type=int, default=128, help='hidden dimension for USE fusion')
    parser.add_argument('--use_num_casm_layers', type=int, default=3, help='number of CASM layers in USE')
    parser.add_argument('--use_num_smpn_layers', type=int, default=3, help='number of SMPN layers in USE')
    parser.add_argument('--caga_hidden_dim', type=int, default=128, help='hidden dimension for CAGA fusion')
    parser.add_argument('--caga_num_heads', type=int, default=8, help='number of attention heads for CAGA')
    parser.add_argument('--clifford_dim', type=int, default=4, help='dimension of Clifford algebra')
    parser.add_argument('--clifford_signature', type=str, default='euclidean', 
                        choices=['euclidean', 'minkowski', 'mixed'], help='signature for Clifford algebra')
    
    # Memory and architecture parameters
    parser.add_argument('--use_memory', action='store_true', default=False, help='whether to use memory for temporal models')
    parser.add_argument('--memory_dim', type=int, default=128, help='dimension of memory')
    parser.add_argument('--output_dim', type=int, default=128, help='output dimension of the model')
    
    # Mamba-specific parameters (for DyGMamba)
    parser.add_argument('--mamba_d_model', type=int, default=128, help='model dimension for Mamba')
    parser.add_argument('--mamba_d_state', type=int, default=16, help='state dimension for Mamba')
    parser.add_argument('--mamba_d_conv', type=int, default=4, help='convolution dimension for Mamba')
    parser.add_argument('--mamba_expand', type=int, default=2, help='expansion factor for Mamba')
    
    # Training and evaluation parameters
    parser.add_argument('--enable_feature_caching', action='store_true', default=True, 
                        help='whether to enable feature caching for efficiency')
    parser.add_argument('--clear_cache_interval', type=int, default=100, 
                        help='interval to clear feature cache during training')
    
    # Backward compatibility aliases
    parser.add_argument('--n_degree', type=int, default=20, help='alias for num_neighbors')
    parser.add_argument('--n_head', type=int, default=2, help='alias for num_heads')
    parser.add_argument('--n_layer', type=int, default=2, help='alias for num_layers')
    parser.add_argument('--drop_out', type=float, default=0.1, help='alias for dropout')
    parser.add_argument('--lr', type=float, default=0.0001, help='alias for learning_rate')
    parser.add_argument('--n_epoch', type=int, default=100, help='alias for num_epochs')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--uniform', action='store_true', default=False, help='whether to use uniform sampling')
    parser.add_argument('--different_new_nodes', action='store_true', default=False, 
                        help='whether to use different new nodes between val and test')


    try:
        args = parser.parse_args()
        args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
        
        # Handle backward compatibility aliases
        if hasattr(args, 'n_degree'):
            args.num_neighbors = args.n_degree
        if hasattr(args, 'n_head'):
            args.num_heads = args.n_head  
        if hasattr(args, 'n_layer'):
            args.num_layers = args.n_layer
        if hasattr(args, 'drop_out'):
            args.dropout = args.drop_out
        if hasattr(args, 'lr'):
            args.learning_rate = args.lr
        if hasattr(args, 'n_epoch'):
            args.num_epochs = args.n_epoch
            
    except:
        parser.print_help()
        sys.exit()

    if args.load_best_configs:
        load_link_prediction_best_configs(args=args)

    return args


def load_link_prediction_best_configs(args: argparse.Namespace):
    """
    load the best configurations for the link prediction task
    :param args: argparse.Namespace
    :return:
    """
    # model specific settings
    if args.model_name == 'TGAT':
        args.num_neighbors = 20
        args.num_layers = 2
        if args.dataset_name in ['enron']:
            args.dropout = 0.2
        else:
            args.dropout = 0.1
        if args.dataset_name in ['reddit']:
            args.sample_neighbor_strategy = 'uniform'
        else:
            args.sample_neighbor_strategy = 'recent'
    elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
        args.num_neighbors = 10
        args.num_layers = 1
        if args.model_name == 'JODIE':
            if args.dataset_name in ['mooc']:
                args.dropout = 0.2
            elif args.dataset_name in ['lastfm']:
                args.dropout = 0.3
            elif args.dataset_name in ['uci']:
                args.dropout = 0.4
            else:
                args.dropout = 0.1
        elif args.model_name == 'DyRep':
            if args.dataset_name in ['mooc', 'lastfm', 'enron', 'uci']:
                args.dropout = 0.0
            else:
                args.dropout = 0.1
        else:
            assert args.model_name == 'TGN'
            if args.dataset_name in ['mooc']:
                args.dropout = 0.2
            elif args.dataset_name in ['lastfm']:
                args.dropout = 0.3
            elif args.dataset_name in ['enron', 'SocialEvo']:
                args.dropout = 0.0
            else:
                args.dropout = 0.1
        if args.model_name in ['TGN', 'DyRep']:
            args.sample_neighbor_strategy = 'recent'
    elif args.model_name == 'CAWN':
        args.time_scaling_factor = 1e-6
        if args.dataset_name in ['mooc', 'SocialEvo', 'uci']:
            args.num_neighbors = 64
        elif args.dataset_name in ['lastfm']:
            args.num_neighbors = 128
        else:
            args.num_neighbors = 32
        args.dropout = 0.1
        args.sample_neighbor_strategy = 'time_interval_aware'
    elif args.model_name == 'TCL':
        args.num_neighbors = 20
        args.num_layers = 2
        if args.dataset_name in ['SocialEvo', 'uci']:
            args.dropout = 0.0
        else:
            args.dropout = 0.1
        if args.dataset_name in ['reddit']:
            args.sample_neighbor_strategy = 'uniform'
        else:
            args.sample_neighbor_strategy = 'recent'
    elif args.model_name == 'GraphMixer':
        args.num_layers = 2
        if args.dataset_name in ['wikipedia']:
            args.num_neighbors = 30
        elif args.dataset_name in ['reddit', 'lastfm']:
            args.num_neighbors = 10
        else:
            args.num_neighbors = 20
        if args.dataset_name in ['wikipedia', 'reddit', 'enron']:
            args.dropout = 0.5
        elif args.dataset_name in ['mooc', 'uci']:
            args.dropout = 0.4
        elif args.dataset_name in ['lastfm']:
            args.dropout = 0.0
        elif args.dataset_name in ['SocialEvo']:
            args.dropout = 0.3
        else:
            args.dropout = 0.1
        args.sample_neighbor_strategy = 'recent'
    elif args.model_name == 'DyGFormer':
        args.num_layers = 2
        if args.dataset_name in ['reddit']:
            args.max_input_sequence_length = 64
            args.patch_size = 2
        elif args.dataset_name in ['mooc', 'enron']:
            args.max_input_sequence_length = 256
            args.patch_size = 4
        elif args.dataset_name in ['lastfm']:
            args.max_input_sequence_length = 512
            args.patch_size = 16
        else:
            args.max_input_sequence_length = 32 
            args.patch_size = 1
        assert args.max_input_sequence_length % args.patch_size == 0
        if args.dataset_name in ['reddit']:
            args.dropout = 0.2
        elif args.dataset_name in ['enron']:
            args.dropout = 0.0
        else:
            args.dropout = 0.1
    elif args.model_name == 'DyGMamba':
        args.num_layers = 2
        if args.dataset_name in ['reddit']:
            args.max_input_sequence_length = 64
            args.patch_size = 1
        elif args.dataset_name in ['mooc', 'enron']:
            args.max_input_sequence_length = 256
            args.patch_size = 1
        elif args.dataset_name in ['lastfm']:
            args.max_input_sequence_length = 128
            args.patch_size = 4
        else:
            args.max_input_sequence_length = 32
            args.patch_size = 1
        assert args.max_input_sequence_length % args.patch_size == 0
        if args.dataset_name in ['reddit']:
            args.dropout = 0.2
        elif args.dataset_name in ['enron']:
            args.dropout = 0.0
        else:
            args.dropout = 0.1
        if args.dataset_name in ['enron']:
            args.max_interaction_times = 30
        elif args.dataset_name in ['mooc','lastfm']:
            args.max_interaction_times = 10
        else:
            args.max_interaction_times = 5
    else:
        raise ValueError(f"Wrong value for model_name {args.model_name}!")

