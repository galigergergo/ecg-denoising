import torch.optim as optim
import torch
import torch.nn as nn
import mlflow
import numpy as np
from tqdm import tqdm
from src.models.FISTANet import FISTANet


EMPTY_LOSS_DICT = {
    'sum': [],
    'disc': [],
    'pred-spars': [],
    'sym': [],
    'spars': [],
    'non-zero': []
}


def l1_loss(pred, target, l1_weight):
    """
    Compute L1 loss;
    l1_weigh default: 0.1
    """
    err = torch.mean(torch.abs(pred - target))
    err = l1_weight * err
    return err


class FISTANetTrainer():
    def __init__(self, Phi, args):
        self.model_name = args['model_name']
        self.fnet_layer_no = args['fnet_layer_no']
        self.fnet_feature_no = args['fnet_feature_no']
        self.device = args['device']
        self.model = FISTANet(self.fnet_layer_no, self.fnet_feature_no).to(self.device)
        self.Phi = Phi
        self.data_dir = args['data_dir']
        self.start_run = args['start_run']
        self.lambda_sp_loss = args['lambda_sp_loss']
        self.lambda_pred_sp_loss = args['lambda_pred_sp_loss']
        self.lambda_sym_loss = args['lambda_sym_loss']
        self.lr = args['lr']
        
        self.batch_size = args['batch_size']
        
        self.lr_dec_after = args['lr_dec_after']
        self.lr_dec_every = args['lr_dec_every']

        # set different lr for regularization weights and network weights
        self.optimizer = optim.Adam([
                {'params': self.model.fcs.parameters()},
                {'params': self.model.w_theta, 'lr': 0.001},
                {'params': self.model.b_theta, 'lr': 0.001},
                {'params': self.model.w_mu, 'lr': 0.001},
                {'params': self.model.b_mu, 'lr': 0.001},
                {'params': self.model.w_rho, 'lr': 0.001},
                {'params': self.model.b_rho, 'lr': 0.001}
            ],
            lr=self.lr, weight_decay=0.001)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.9)  # step-wise

        self.save_path = args['save_path']
        self.multi_gpu = args['multi_gpu']
        self.device = args['device']
        self.log_interval = args['log_interval']
        self.test_epoch = args['test_epoch']
        self.train_loss = nn.MSELoss()


    def preprocess_batch(self, x_in, y_target, x_0):
        # measured vector (104*1); add channels
        # CIKK: vector b (16) --\/
        x_in = torch.unsqueeze(x_in, 2)

        # target image (64*64)
        y_target = torch.unsqueeze(y_target, 2)

        # System matrix
        Phi = self.Phi.repeat((x_in.shape[0], 1, 1))

        # initial image from one-step inversion
        # CIKK: initialization (16) --\/
        x_0 = torch.from_numpy(np.zeros((x_in.shape[0], self.Phi.shape[1])))
        x_0 = torch.unsqueeze(x_0, 2)
        # x = Ab initialization
        # x_0 = torch.bmm(Phi.permute(0, 2, 1), x_in)
        
        x_0 = x_0.clone().detach().to(device=self.device)
        x_in = x_in.clone().detach().to(device=self.device)
        y_target = y_target.clone().detach().to(device=self.device)

        return x_in, x_0, y_target, Phi
    

    def loss_dssps(self, pred, y_target, pred_alph, loss_sym, loss_st, losses_dict):
        # Discrepancy loss
        loss_discrepancy_1 = self.train_loss(pred, y_target)
        loss_discrepancy_2 = l1_loss(pred, y_target, 0.1)
        loss_discrepancy = loss_discrepancy_1 + loss_discrepancy_2
        
        # Alpha sparsity loss - L1 norm for alpha after max-abs normalization
        mins = pred_alph.squeeze().min(dim=1).values.repeat((pred_alph.squeeze().shape[1], 1)).T
        maxs = pred_alph.squeeze().max(dim=1).values.repeat((pred_alph.squeeze().shape[1], 1)).T
        absmax = torch.stack([mins, maxs]).abs().max(dim=0).values
        loss_pred_sparcity = torch.mean(torch.abs(pred_alph.squeeze() / absmax))
        
        # Symmetry of F and F^{-1} loss
        loss_constraint = 0
        for k, _ in enumerate(loss_sym, 0):
           loss_constraint += torch.mean(torch.pow(loss_sym[k], 2))

        # Sparsity in F space loss
        sparsity_constraint = 0
        for k, _ in enumerate(loss_st, 0):
            sparsity_constraint += torch.mean(torch.abs(loss_st[k]))

        # Final loss as the weighted some of single losses
        loss = loss_discrepancy + self.lambda_sym_loss * loss_constraint + self.lambda_sp_loss * sparsity_constraint + self.lambda_pred_sp_loss * loss_pred_sparcity

        # Append losses to per epoch global list
        losses_dict['sum'].append(loss.item())
        losses_dict['disc'].append(loss_discrepancy.item())
        losses_dict['pred-spars'].append(self.lambda_pred_sp_loss * loss_pred_sparcity.item())
        losses_dict['sym'].append(self.lambda_sym_loss * loss_constraint.item())
        losses_dict['spars'].append(self.lambda_sp_loss * sparsity_constraint.item())
        losses_dict['non-zero'].append((torch.sum(pred_alph.abs()>1e-5) / (pred_alph.shape[0] * pred_alph.shape[1])).item())

        return loss, losses_dict
    
    
    def train(self, train_loader, valid_loader, epochs, start_epoch=0):
        # if start_epoch:
        #     self.load_model(self.start_epoch)

        for epoch in tqdm(range(1 + start_epoch, epochs + start_epoch + 1)):
            losses_dict = EMPTY_LOSS_DICT.copy()

            self.model.train(True)
            for batch_idx, (x_in, y_target, x_0, x_bpdn) in enumerate(train_loader):
                x_in, x_0, y_target, Phi = self.preprocess_batch(x_in, y_target, x_0)

                self.model.zero_grad(set_to_none=True)
                # self.optimizer.zero_grad()

                pred_alph, loss_sym, loss_st = self.model(x_0, x_in-y_target, Phi)   # forward
                pred = x_in - torch.bmm(Phi, pred_alph)

                loss, losses_dict = self.loss_dssps(pred, y_target, pred_alph, loss_sym, loss_st, losses_dict)

                # backpropagate the gradients
                loss.backward()
                self.optimizer.step()
                
            # log average epoch loss values to MLflow
            for k, v in losses_dict.items():
                mlflow.log_metric('loss_train_' + k, np.mean(v), step=epoch)

            losses_dict = EMPTY_LOSS_DICT.copy()
            self.model.eval()
            with torch.no_grad():
                for batch_idy, (x_in, y_target, x_0, x_bpdn) in enumerate(valid_loader):
                    x_in, x_0, y_target, Phi = self.preprocess_batch(x_in, y_target, x_0)
                    
                    pred_alph, loss_sym, loss_st = self.model(x_0, x_in-y_target, Phi)   # forward
                    pred = x_in - torch.bmm(Phi, pred_alph)
                    x0_pred = x_in - torch.bmm(Phi, x_0)

                    loss, losses_dict = self.loss_dssps(pred, y_target, pred_alph, loss_sym, loss_st, losses_dict)
                    
                    # plot validation batch TODO: save as MLflow artifact
                    # if not epoch % 10 and not batch_idy:
                    #     test_plot_est(x_in, x0_pred, pred, y_target, self.save_path, 'valid_ep%d_btch%d.png' % (epoch, batch_idy))
            
                # log average epoch loss values to MLflow
                for k, v in losses_dict.items():
                    mlflow.log_metric('loss_valid_' + k, np.mean(v), step=epoch)
        
        save_every = 10        # save model ever N-th epoch
        if not (epoch % save_every) and epoch > 0:
            # with torch.no_grad():
            #     model_signature = mlflow.models.infer_signature(X_train.numpy(), self.model(X_train).numpy())
            mlflow.pytorch.log_model(self.model, artifact_path=f'models/AE_ep{epoch}')#, signature=model_signature)


    def evaluate(self, test_loader, criterion=None, crit_text=None):
        assert not ((type(criterion) != type(None)) ^ (crit_text != None))
        losses_dict = EMPTY_LOSS_DICT.copy()
        self.model.eval()
        with torch.no_grad():
            for batch_idy, (x_in, y_target, x_0, x_bpdn) in enumerate(test_loader):
                x_in, x_0, y_target, Phi = self.preprocess_batch(x_in, y_target, x_0)
                
                pred_alph, loss_sym, loss_st = self.model(x_0, x_in-y_target, Phi)   # forward
                pred = x_in - torch.bmm(Phi, pred_alph)

                if type(criterion) == type(None):
                    loss, losses_dict = self.loss_dssps(pred, y_target, pred_alph, loss_sym, loss_st, losses_dict)
                    for k, v in losses_dict.items():
                        mlflow.log_metric('loss_test_' + k, np.mean(v))
                else:
                    loss = criterion(pred, y_target)
                    mlflow.log_metric(f'loss_test_{crit_text}', loss.item())
                
                # plot validation batch TODO: save as MLflow artifact
                # if not epoch % 10 and not batch_idy:
                #     test_plot_est(x_in, x0_pred, pred, y_target, self.save_path, 'valid_ep%d_btch%d.png' % (epoch, batch_idy))
