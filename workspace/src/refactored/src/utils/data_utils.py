import torch
import os
import pandas as pd
import mlflow
from typing import Union


def read_mlflow_dataset(data_path: str, data_date: str, context: str, targets: str, device: str) -> Union[torch.Tensor, torch.Tensor]:
    '''
    Reads CSV dataset, logs it to MLflow and returns X and y PyTorch tensors.
    '''
    assert context in ['train', 'validate', 'test']
    
    file_path = os.path.join(data_path, 'data', 'OSM_road_network', data_date, f'BP_safety-network_{data_date}_NN_{context}.csv')
    df = pd.read_csv(file_path)
    # Move accident number columns to the right side of the dataframe 
    df = df[list(df.columns[~df.columns.str.contains(pat = 'acc_no_')]) + list(df.columns[df.columns.str.contains(pat = 'acc_no_')])]
    non_accident_dim = len(list(df.columns[~df.columns.str.contains(pat = 'acc_no_')]))
    
    dataset = mlflow.data.from_pandas(pd.DataFrame(df), source=file_path, name=f'BP_safety-network_{data_date}_NN_{context}', targets=targets)
    mlflow.log_input(dataset, context=context)

    return torch.tensor(df.drop(targets, axis=1).values, device=device).float(), torch.tensor(df[targets].values, device=device).float(), non_accident_dim
