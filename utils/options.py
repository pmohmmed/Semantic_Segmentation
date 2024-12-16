import argparse

def str_to_bool(value):
    if value.lower() in ['true', '1', 't', 'y', 'yes']:
        return True
    elif value.lower() in ['false', '0', 'f', 'n', 'no']:
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def train_opt():
    parser = argparse.ArgumentParser(description="Train a model on a dataset.")

    ## requried
    parser.add_argument('--enc', type=str_to_bool, required=True, help="Encode labels (True or False)")
    parser.add_argument('--data_path', type=str, required=True, help="Dataset path.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to save the model.")
    ## not required
    parser.add_argument('--pre_obj_path', type=str, default='data/pre.pkl', help="Path to save the preprocessor.")
    parser.add_argument('--res', type=int, default=256, help="Image Resulotion")
    parser.add_argument('--mods', action='store_true', help="Model Summary")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs.")
    parser.add_argument('--lr', type=float, default=0.0005, help="Learning rate.")

    return parser.parse_args()


def test_opt():
    parser = argparse.ArgumentParser(description="Inference using test data.")
    
    ## required
    parser.add_argument('--model_path', type=str, required=True, help="Model weights path.")
    parser.add_argument('--data_path', type=str, required=True, help="Dataset path.")
    parser.add_argument('--results_path', type=str, required=True, help="Path to save the results.")
    parser.add_argument('--pre_obj_path', type=str, required=True, help="Preprocessor path.")
    parser.add_argument('--show_results', type=str_to_bool, required=True, help="Show input imgs with it's predicted masks.")

    return parser.parse_args()

def aug_opt():
    parser = argparse.ArgumentParser(description="Data Augmentation.")
    
    ## required
    parser.add_argument('--pre', type=str_to_bool, required=True, help="Apply preprocessing before saving.")
    parser.add_argument('--data_path', type=str, required=True, help="Dataset path.")
    parser.add_argument('--aug_data_path', type=str, required=True, help="Path to save the new data.")
    ## not required
    parser.add_argument('--res', type=int, default=256, help="Image Resulotion, in case you applied preprocessing")
    parser.add_argument('--pre_obj_path', type=str, default='data/pre.pkl', help="Path to save the preprocessor.")

    return parser.parse_args()


