import copy
from collections import OrderedDict

import torch
import numpy as np


class Lca:
    # TODO: introduce with
    # with lca.set_theta():
    #     optimizer.step()

    def __init__(self, model, criterion, device):
        """
        Parameters
        ----------
        model_org: torch.nn.Module
            model to set parameters, and calculate loss/gradient.
        criterion: torch.nn.Module
            loss function
        """
        self.model = copy.deepcopy(model)
        self.criterion = copy.deepcopy(criterion)
        self.device = device
        self.theta_list = [None, None, None]

    def set_cur_theta(self, model):
        # name_param_dict = OrderedDict()
        # for n, p in model.named_parameters():
        # name_param_dict[n] = p.data
        self.theta_list[0] = copy.deepcopy(model.state_dict(keep_vars=False))

    def set_next_theta(self, model):
        self.theta_list[-1] = copy.deepcopy(model.state_dict(keep_vars=False))

    def set_fractional_theta(self):
        if self.theta_list[0] is None:
            raise ValueError('Current parameters must be set')
        if self.theta_list[-1] is None:
            raise ValueError('Current parameters must be set')

        theta_frac = OrderedDict()
        for (n1, p1), (n2, p2) in zip(self.theta_list[0].items(), self.theta_list[-1].items()):
            if n1 != n2:
                raise ValueError(
                    'Names of current and next parameter are different')
            # 1/2 theta_t + 1/2 theta_(t+1)
            theta_frac[n1] = (p1.data + p2.data) / 2

        self.theta_list[1] = theta_frac

    def calc_grad(self, loader):
        """
        Calculate LCA for each layer.

        Parameters
        ----------
        x: torch.Tensor
            input of model
        y: torch.Tensor
            output of model

        Returns
        -------
        OrderedDict
            key: parameter name. e.g. conv1.weight
            vlaue: LCA for each parameter
        """

        lca = OrderedDict()
        coeffs = [1, 4, 1]

        # set 1/2 theta_t + 1/2 theta_(t+1)
        self.set_fractional_theta()

        # initialize lca
        for n, p in self.model.named_parameters():
            lca[n] = torch.zeros(*p.size()).to(self.device)

        n_batches = len(loader)
        # record loss change
        # L(theta_t): loss_vals[i, 0], L(theta_(t+1)): loss_vals[i, -1]
        loss_vals = np.zeros((n_batches, 3))

        for idx, (theta, coeff) in enumerate(zip(self.theta_list, coeffs)):
            # set parameter to model
            self.model.load_state_dict(theta)

            self.model.zero_grad()
            for b_idx, data in enumerate(loader):
                img, label = data
                img, label = img.to(self.device), label.to(self.device)

                logit = self.model(img)
                loss = self.criterion(logit, label)
                # accumulate gradient
                loss.backward()

                loss_vals[b_idx, idx] = loss.item()

            # calculate LCA
            # coeff * delta_L(theta) / 6 / n_repeats
            for n, p in self.model.named_parameters():
                if p is not None and p.grad is not None:
                    lca[n] += coeff * p.grad.data / sum(coeffs) / n_batches

        loss_change = (loss_vals[:, -1] - loss_vals[:, 0]).mean(axis=0)
        print('loss change: %.6f' % loss_change)
        print(loss_vals)

        # inner product <delta_L(theta), theta_(t+1) - theta_t>
        for k, v in lca.items():
            lca[k] *= (self.theta_list[-1][k] - self.theta_list[0][k])

        return lca
