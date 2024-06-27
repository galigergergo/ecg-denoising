import torch
import mlflow
import numpy as np
from tqdm import tqdm
from src.models.AEFCN import AE, FC, AEFCN


class AETrainer():
    def __init__(self, inp_dim, noise_factor, enc_dim=4, learning_rate=1e-3, weight_decay=1e-8):
        self.inp_dim = inp_dim
        self.noise_factor = noise_factor
        self.enc_dim = enc_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.model = AE(self.inp_dim, self.enc_dim, self.noise_factor)
        self.criterion = torch.nn.MSELoss()
        self.crit_text = 'mse'
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    
    def train(self, X_train, y_train, X_valid, y_valid, epochs):
        X_train, X_valid = X_train[:, :self.inp_dim], X_valid[:, :self.inp_dim]
        y_train = (X_train + self.noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)).float()
        y_valid = (X_valid + self.noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_valid.shape)).float()
        
        with torch.no_grad():
            model_signature = mlflow.models.infer_signature(X_train.numpy(), self.model(X_train).numpy())

        for epoch in tqdm(range(epochs)):
            self.optimizer.zero_grad()
            
            reconstr = self.model(X_train)
            
            loss = self.criterion(reconstr, y_train)
            loss.backward()
            
            self.optimizer.step()

            mlflow.log_metric('loss_mse_train', loss.item(), step=epoch)

            # validation
            with torch.no_grad():
                reconstr = self.model(X_valid)
                loss = self.criterion(reconstr, y_valid)
                mlflow.log_metric('loss_mse_validate', loss.item(), step=epoch)

        mlflow.pytorch.log_model(self.model, artifact_path=f'models/AE_ep{epoch}', signature=model_signature)

    
    def evaluate(self, X_test, y_test, criterion=None, crit_text=None):
        assert not ((criterion != None) ^ (crit_text != None))
        X_test = X_test[:, :self.inp_dim]
        y_test = (X_test + self.noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)).float()
        pred = self.model(X_test)
        if criterion == None:
            loss = self.criterion(pred, y_test)
            mlflow.log_metric(f'loss_{self.crit_text}_test', loss.item())
        else:
            loss = criterion(pred, y_test)
            mlflow.log_metric(f'loss_{crit_text}_test', loss.item())


class AEFCNTrainer():
    def __init__(self, autoencoder, inp_dim, learning_rate=1e-3):
        self.ae = autoencoder

        self.inp_dim = inp_dim
        self.learning_rate = learning_rate
        self.fc = FC(inp_dim)
        self.criterion = torch.nn.MSELoss()
        self.crit_text = 'mse'
        self.optimizer = torch.optim.Adam(self.fc.parameters(), lr=self.learning_rate)
        
    
    def train(self, X_train, y_train, X_valid, y_valid, epochs):
        # Split data for the autoencoder
        X_train_ae, X_valid_ae = X_train[:, :self.ae.inp_dim], X_valid[:, :self.ae.inp_dim]

        # Encode data with autoencoder
        X_train_enc = self.ae.encode(X_train_ae).detach()
        X_valid_enc = self.ae.encode(X_valid_ae).detach()

        # Concatenate encoded data with previous year accident data
        X_train_fc = torch.cat((X_train_enc, X_train[:, self.ae.inp_dim:]), 1)
        X_valid_fc = torch.cat((X_valid_enc, X_valid[:, self.ae.inp_dim:]), 1)

        # Train and validate FC network
        for epoch in tqdm(range(epochs)):
            self.optimizer.zero_grad()
            
            reconstr = self.fc(X_train_fc)
            
            loss = self.criterion(reconstr, y_train)
            loss.backward()
            
            self.optimizer.step()

            mlflow.log_metric('loss_mse_train', loss.item(), step=epoch)

            # Validation
            with torch.no_grad():
                reconstr = self.fc(X_valid_fc)
                loss = self.criterion(reconstr, y_valid)
                mlflow.log_metric('loss_mse_validate', loss.item(), step=epoch)

        # Define AEFCN model from trained AE and FC networks
        aefcn_model = AEFCN(self.ae, self.fc)
        self.model = aefcn_model
        
        # Infer model signature for MLflow
        with torch.no_grad():
            model_signature = mlflow.models.infer_signature(X_train.numpy(), aefcn_model(X_train).numpy())
        
        # Log model as an MLflow artifact
        mlflow.pytorch.log_model(self.model, artifact_path=f'models/AEFCN_ep{epoch}', signature=model_signature)
    
    def evaluate(self, X_test, y_test, criterion=None, crit_text=None):
        assert not ((criterion != None) ^ (crit_text != None))
        pred = self.model(X_test)
        if criterion == None:
            loss = self.criterion(pred, y_test)
            mlflow.log_metric(f'loss_{self.crit_text}_test', loss.item())
        else:
            loss = criterion(pred, y_test)
            mlflow.log_metric(f'loss_{crit_text}_test', loss.item())
