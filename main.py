from gcn import GCN 
from helpers import load_data
import pathlib 

from typing import Union, List

import sys 

def run_with_dataset(directory: Union[str, 'pathlib.Path'], 
                     dataset: str, 
                     hidden: List[int] = [16], 
                     dropout: float = 0.5, 
                     learning_rate: float = 0.01, 
                     weight_decay: float = 5e-4,
                     epochs: int = 200,
                     verbose: bool = True) -> None:
    """Runs training with a given dataset

    Args:
        directory: Path to datasets 
        dataset: dataset to run on 
        hidden: Hidden Layer sizes
        dropout: Dropout Rate
        learning_rate: Learning Rate 
        weight_decay: Weight decay
        epochs: Number of epochs to train for
        verbose: If True, prints messages during training time. \
            Defaults to true
    """
    gcn = GCN(*load_data(directory, dataset))
    gcn.train(hidden=hidden, 
              dropout=dropout, 
              learning_rate=learning_rate, 
              weight_decay=weight_decay,
              epochs=epochs,
              verbose=verbose)
    return gcn

if __name__ == '__main__':
    import sys 

    if len(sys.argv )< 3:
        print(f'Usage: python {sys.argv[0]} <path/to/data> <dataset>')
        sys.exit(1)
        
    run_with_dataset(*sys.argv[1:])