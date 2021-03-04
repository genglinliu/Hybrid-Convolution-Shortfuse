import pandas as pd
import torch


def get_covariates(cov_name, split):
    """
    get a list of strutured covariates as a pytorch tensor
    split by train/val/test
    
    input:
     - cov_name = 'Male'
     - split = 'train', 'val' or 'test'
    
    Here we're mostly interested in gender so our covaraite is male/female
    Male = 1, Female = 0
    
    return:
     - a pytorch tensor, list of gender attribute values
    
    """
    
    list_attr_fn = "../data/celeba/list_attr_celeba.txt"
    splits_fn = "../data/celeba/list_eval_partition.txt"
    attr = pd.read_csv(list_attr_fn, delim_whitespace=True, header=1)
    splits = pd.read_csv(splits_fn, delim_whitespace=True, header=None, index_col=0)

    attr = (attr + 1) // 2  # map from {-1, 1} to {0, 1}
    
    train_mask = (splits[1] == 0)
    val_mask = (splits[1] == 1)
    test_mask = (splits[1] == 2)
    
    
    if split == 'train':
        return torch.as_tensor(attr[cov_name][train_mask])
    elif split == 'val':
        return torch.as_tensor(attr[cov_name][val_mask])
    else:
        return torch.as_tensor(attr[cov_name][test_mask])



# def main():
#     res = get_covariates('Male', 'train')
#     print(res)
    
# if __name__ == "__main__":
#     main()