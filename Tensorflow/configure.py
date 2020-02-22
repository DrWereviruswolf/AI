import argparse
import sys


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()

    # Data loading params
    parser.add_argument("--k_fold", default=3, type=int, help="K of k-fold")
    parser.add_argument("--av_data_path", default="data_2d_tuple/3fold_for_attribute_value/",
                        type=str, help="Path of attribute&value data")
    parser.add_argument("--ta_data_path", default="data_2d_tuple/3fold_for_time_attribute/",
                        type=str, help="Path of time&attribute data")
    parser.add_argument("--max_sentence_length", default=154,
                        type=int, help="Max sentence length in data")

    # Model Hyper-parameters
    parser.add_argument("--sequence_length", default=154,
                        type=int, help="Length of one sentence")
    parser.add_argument("--vocab_size", default=2840,
                        type=int, help="Size of vocabulary")
    # Embeddings
    parser.add_argument("--embedding_path", default=None,
                        type=str, help="Path of pre-trained word embeddings (glove)")
    parser.add_argument("--embedding_dim", default=100,
                        type=int, help="Dimensionality of word embedding (default: 100)")
    parser.add_argument("--emb_dropout_keep_prob", default=0.7,
                        type=float, help="Dropout keep probability of embedding layer (default: 0.7)")
    # AttLSTM
    parser.add_argument("--hidden_size", default=100,
                        type=int, help="Dimensionality of RNN hidden (default: 100)")
    parser.add_argument("--rnn_dropout_keep_prob", default=0.7,
                        type=float, help="Dropout keep probability of RNN (default: 0.7)")

    # Misc
    parser.add_argument("--dropout_keep_prob", default=0.5,
                        type=float, help="Dropout keep probability of output layer (default: 0.5)")
    parser.add_argument("--l2_reg_lambda", default=1e-5,
                        type=float, help="L2 regularization lambda (default: 1e-5)")

    # Training parameters
    parser.add_argument("--batch_size", default=800,
                        type=int, help="Batch Size (default: 512)")
    parser.add_argument("--num_epochs", default=50,
                        type=int, help="Number of training epochs (Default: 100)")
    parser.add_argument("--final_epochs_av", default=27,
                        type=int, help="Number of training epochs (Default: 27/1230steps)")
    parser.add_argument("--final_epochs_ta", default=62,
                        type=int, help="Number of training epochs (Default: 62/1740steps)")
    parser.add_argument("--display_every", default=15,
                        type=int, help="Number of iterations to display training information")
    parser.add_argument("--evaluate_every", default=45,
                        type=int, help="Evaluate model on dev set after this many steps (default: 125)")
    parser.add_argument("--num_checkpoints", default=5,
                        type=int, help="Number of checkpoints to store (default: 5)")
    parser.add_argument("--learning_rate", default=1.0,
                        type=float, help="Which learning rate to start with (Default: 1.0)")
    parser.add_argument("--decay_rate", default=0.9,
                        type=float, help="Decay rate for learning rate (Default: 0.9)")

    # Testing parameters
    parser.add_argument("--checkpoint_dir", default="params",
                        type=str, help="Checkpoint directory from training run")

    # Misc Parameters
    parser.add_argument("--allow_soft_placement", default=True,
                        type=bool, help="Allow device soft device placement")
    parser.add_argument("--log_device_placement", default=False,
                        type=bool, help="Log placement of ops on devices")
    parser.add_argument("--gpu_allow_growth", default=True,
                        type=bool, help="Allow gpu memory growth")

    if len(sys.argv) == 0:
        parser.print_help()
        sys.exit(1)

    print("")
    args = parser.parse_args()
    for arg in vars(args):
        print("{}={}".format(arg.upper(), getattr(args, arg)))
    print("")

    return args


FLAGS = parse_args()
