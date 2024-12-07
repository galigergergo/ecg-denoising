import torch.optim as optim
import torch
import torch.nn as nn
import mlflow
import numpy as np
from tqdm import tqdm
from src.utils.plotting_utils import plot_est_comp, plot_alpha_comp


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


class DCDicLTrainer():
    def __init__(self, model, Phi, bpdn_est, dictionary, args):        
        self.model = model
        self.Phi = Phi
        self.bpdn_est = bpdn_est
        self.dictionary = dictionary

        self.device = args['device']

        self.load_model_run = args['load_model_run']
        self.load_model_epoch = args['load_model_epoch']
        
        self.batch_size = args['batch_size']
        self.lr = args['lr']        
        self.lr_dec_after = args['lr_dec_after']
        self.lr_dec_every = args['lr_dec_every']

        if self.load_model_epoch != None:
            self.model = mlflow.pytorch.load_model(f'runs:/{self.load_model_run}/models/FISTA-Net_ep{self.load_model_epoch}')

        self.device = args['device']
        self.train_loss = nn.MSELoss()


    def preprocess_batch(self, y_in, y_target, x_0, x_bpdn):
        repeat_dim = 5
        
        # measured vector: gt + noise
        y_in = torch.unsqueeze(y_in, 2)
        y_in = y_in.repeat(1, 1, repeat_dim)
        y_in = torch.unsqueeze(y_in, 1).float()

        # target vector: gt
        y_target = torch.unsqueeze(y_target, 2)
        y_target = torch.tile(y_target, (1, 1, repeat_dim))
        y_target = torch.unsqueeze(y_target, 1).float()

        # System matrix
        # Phi = self.Phi.repeat((x_in.shape[0], 1, 1))

        # initial image from one-step inversion
        # x_0 = torch.from_numpy(np.zeros((x_in.shape[0], self.Phi.shape[1])))
        # x_0 = torch.unsqueeze(x_0, 2)
        # x = Ab initialization
        # x_0 = torch.bmm(Phi.permute(0, 2, 1), x_in)

        # x_bpdn = torch.unsqueeze(x_bpdn, 2)
        
        # x_0 = x_0.clone().detach().to(device=self.device)
        # x_in = x_in.clone().detach().to(device=self.device)
        # y_target = y_target.clone().detach().to(device=self.device)

        return y_in, y_target
    

    def loss_dssps(self, pred, y_target, pred_alph, loss_sym, loss_st, losses_dict):
        # Discrepancy loss
        loss_discrepancy_1 = self.train_loss(pred, y_target)
        loss_discrepancy_2 = l1_loss(pred, y_target, 0.1)
        loss_discrepancy = loss_discrepancy_1 + loss_discrepancy_2
        
        # Alpha sparsity loss - L1 norm for alpha after max-abs normalization
        mins = pred_alph.squeeze().min(dim=1).values.repeat((pred_alph.squeeze().shape[1], 1)).T
        maxs = pred_alph.squeeze().max(dim=1).values.repeat((pred_alph.squeeze().shape[1], 1)).T
        absmax = torch.stack([mins, maxs]).abs().max(dim=0).values
        # loss_pred_sparcity = torch.square(torch.mean(torch.abs(pred_alph.squeeze() / absmax)))
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
        losses_dict['non-zero'].append((torch.sum(pred_alph.abs()>1e-3) / (pred_alph.shape[0] * pred_alph.shape[1])).item())

        return loss, losses_dict
    
    
    def train(self, train_loader, valid_loader, epochs, start_epoch=0, log_model_every=10, log_comp_fig_every=10):
        # if start_epoch:
        #     self.load_model(self.start_epoch)
        
        for epoch in tqdm(range(1 + start_epoch, epochs + start_epoch + 1)):
            # losses_dict = {k: [] for k in EMPTY_LOSS_DICT.keys()}

            # self.model.train(True)
            for batch_idx, (y_in, y_target, x_0, x_bpdn) in enumerate(train_loader):
                y_in, y_target = self.preprocess_batch(y_in, y_target, x_0, x_bpdn)
                
                sigma = torch.Tensor([25.0]).repeat(self.batch_size, 1).unsqueeze(2).unsqueeze(3)

                self.model.feed_data_2(y_in, y_target, sigma)

                self.model.train()
                
                logger = None
                self.model.log_train_2(batch_idx, epoch, logger)
                
            # # log average epoch loss values to MLflow
            # for k, v in losses_dict.items():
            #     mlflow.log_metric('loss_train_' + k, np.mean(v), step=epoch)

#             losses_dict = {k: [] for k in EMPTY_LOSS_DICT.keys()}
#             self.model.eval()
#             with torch.no_grad():
#                 for batch_idy, (x_in, y_target, x_0, x_bpdn) in enumerate(valid_loader):
#                     x_in, x_0, y_target, x_bpdn, Phi = self.preprocess_batch(x_in, y_target, x_0, x_bpdn)
                    
#                     pred_alph, loss_sym, loss_st = self.model(x_0, x_in-y_target, Phi)   # forward
#                     pred = x_in - torch.bmm(Phi, pred_alph)
#                     x0_pred = x_in - torch.bmm(Phi, x_0)

#                     loss, losses_dict = self.loss_dssps(pred, y_target, pred_alph, loss_sym, loss_st, losses_dict)
                    
#                     # plot validation batch
#                     if not epoch % log_comp_fig_every and not batch_idy:
#                         plot_est_comp(x_in, x0_pred, pred, y_target, self.bpdn_est, self.dictionary,
#                                       'valid', epoch, batch_idy)
#                         plot_alpha_comp(x_0, pred_alph, x_bpdn, 'valid', epoch, batch_idy)
            
#                 # log average epoch loss values and model parameters to MLflow
#                 for k, v in losses_dict.items():
#                     mlflow.log_metric('loss_valid_' + k, np.mean(v), step=epoch)
#                 for k, v in {'w_theta': self.model.w_theta.item(), 'b_theta': self.model.b_theta.item(),
#                              'w_mu': self.model.w_mu.item(), 'b_mu': self.model.b_mu.item(),
#                              'w_rho': self.model.w_rho.item(), 'b_rho': self.model.b_rho.item()}.items():
#                     mlflow.log_metric(k, v, step=epoch)
        
#             if not (epoch % log_model_every) and epoch > 0:
#                 # with torch.no_grad():
#                 #     model_signature = mlflow.models.infer_signature(X_train.numpy(), self.model(X_train).numpy())
#                 mlflow.pytorch.log_model(self.model, artifact_path=f'models/FISTA-Net_ep{epoch}')#, signature=model_signature)


    def evaluate(self, test_loader, criterion=None, crit_text=None):
        assert not ((type(criterion) != type(None)) ^ (crit_text != None))
        losses_dict = {k: [] for k in EMPTY_LOSS_DICT.keys()}
        self.model.eval()
        with torch.no_grad():
            for batch_idy, (x_in, y_target, x_0, x_bpdn) in enumerate(test_loader):
                x_in, x_0, y_target, x_bpdn, Phi = self.preprocess_batch(x_in, y_target, x_0, x_bpdn)
                
                pred_alph, loss_sym, loss_st = self.model(x_0, x_in-y_target, Phi)   # forward
                pred = x_in - torch.bmm(Phi, pred_alph)

                if type(criterion) == type(None):
                    loss, losses_dict = self.loss_dssps(pred, y_target, pred_alph, loss_sym, loss_st, losses_dict)
                    for k, v in losses_dict.items():
                        mlflow.log_metric('loss_test_' + k, np.mean(v))
                else:
                    loss = criterion(pred, y_target)
                    mlflow.log_metric(f'loss_test_{crit_text}', loss.item())


    def infer(self, test_loader):
        res = []
        self.model.eval()
        with torch.no_grad():
            for batch_idy, (x_in, y_target, x_0, x_bpdn) in enumerate(test_loader):
                x_in, x_0, y_target, x_bpdn, Phi = self.preprocess_batch(x_in, y_target, x_0, x_bpdn)
                
                pred_alph, loss_sym, loss_st = self.model(x_0, x_in-y_target, Phi)   # forward
                pred = x_in - torch.bmm(Phi, pred_alph)

                res.append([pred, pred_alph])
        return res