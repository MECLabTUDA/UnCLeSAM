# -- From: https://github.com/MECLabTUDA/Lifelong-nnUNet/blob/f22c01eebc2b0c542ca0a40ac722f90aab05fc54/nnunet_ext/training/loss_functions/deep_supervision.py#L86 -- #
class RWalkLoss:
    """
    RWalk Loss.
    """
    def __init__(self, ewc_lambda=0.4, fisher=dict(), params=dict(), parameter_importance=dict()):
        self.ewc_lambda = ewc_lambda
        self.tasks = list(fisher.keys())[:-1]   # <-- Current task is already in there not like simple EWC!
        self.fisher = fisher
        self.params = params
        self.parameter_importance = parameter_importance

    def loss(self, network_params):
        # -- Update the network_params -- #
        loss = 0
        for task in self.tasks:
            for name, param in network_params: # Get named parameters of the current model
                # -- Extract corresponding fisher and param values -- #
                param_ = param
                fisher_value = self.fisher[task][name]
                param_value = self.params[task][name]
                importance = self.parameter_importance[task][name]
                
                # -- loss = loss_{t} + ewc_lambda * \sum_{i} (F_{i} + S(param_{i})) * (param_{i} - param_{t-1, i})**2 -- #
                loss += self.ewc_lambda * ((fisher_value + importance) * (param_ - param_value).pow(2)).sum()
        return loss

# -- From: https://github.com/MECLabTUDA/Lifelong-nnUNet/blob/f22c01eebc2b0c542ca0a40ac722f90aab05fc54/nnunet_ext/training/loss_functions/deep_supervision.py#L15 -- #
class EWCLoss:
    """
    EWC Loss.
    """
    def __init__(self, ewc_lambda=0.4, fisher=dict(), params=dict()):
        self.ewc_lambda = ewc_lambda
        self.tasks = list(fisher.keys())
        self.fisher = fisher
        self.params = params

    def loss(self, network_params):
        # -- Update the network_params -- #
        loss = 0
        for task in self.tasks:
            for name, param in network_params: # Get named parameters of the current model
                # -- Extract corresponding fisher and param values -- #
                # fisher_value = self.fisher[task][name]
                # param_value = self.params[task][name]
                param_ = param
                fisher_value = self.fisher[task][name]
                param_value = self.params[task][name]
                # loss = to_cuda(loss, gpu_id=param.get_device())
                
                # -- loss = loss_{t} + ewc_lambda/2 * \sum_{i} F_{i}(param_{i} - param_{t-1, i})**2 -- #
                loss += self.ewc_lambda/2 * (fisher_value * (param_ - param_value).pow(2)).sum()
        return loss