from utils.util import AverageMeter
import torch 
from clients.client_base import Client

class ClientFedAvg(Client):
    def __init__(self, args, client_idx, is_corrupted):
        super().__init__(args, client_idx, is_corrupted)        
        
    def train(self):
        trainloader = self.load_train_data()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.wd)
        self.model = self.model.to(self.device)
        self.model.train()
        losses = AverageMeter()
        accs = AverageMeter()

        for e in range(self.local_epochs):
            for i, (x, y) in enumerate(trainloader):
                x = x.to(self.device)
                y = y.to(self.device)

                # forward pass
                output = self.model(x)
                loss = self.loss(output, y)

                acc = (output.argmax(1) == y).float().mean() * 100.0
                accs.update(acc, x.size(0))
                losses.update(loss.item(), x.size(0))

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        self.model = self.model.to("cpu")
        return accs.avg, losses.avg

    def get_update(self, global_model):
        """
        Calculates the update vector (difference).
        (Implementation from previous response is likely correct, ensure it includes traceback)
        """
        update = torch.tensor([], dtype=torch.float32, device="cpu")
        self.model.eval()
        global_model.eval()
        try:
            with torch.no_grad():
                local_params = self.model.state_dict()
                global_params = global_model.state_dict()
                for name, global_param in global_model.named_parameters():
                    if not global_param.requires_grad: continue
                    if name in local_params:
                        local_param = local_params[name].to("cpu")
                        global_param_cpu = global_param.detach().clone().to("cpu")
                        delta = local_param - global_param_cpu
                        if torch.isnan(delta).any() or torch.isinf(delta).any():
                            # Use logging if available, otherwise print
                            log_func = getattr(self, 'logger.warning', print)
                            log_func(f"Warning: NaN/Inf detected in delta for param {name} in client {self.client_idx}. Appending zeros.")
                            delta = torch.zeros_like(delta)
                        update = torch.cat((update, delta.view(-1)))
                    else:
                        log_func = getattr(self, 'logger.warning', print)
                        log_func(f"Warning: Parameter {name} not found in local model during get_update for client {self.client_idx}. Appending zeros.")
                        update = torch.cat((update, torch.zeros(global_param.numel(), dtype=torch.float32, device="cpu")))
        except Exception as e:
            log_func = getattr(self, 'logger.error', print)
            log_func(f"ERROR calculating update for client {self.client_idx}: {e}")
            # Consider adding traceback print here if not using logger.exception
            traceback.print_exc()
            return torch.tensor([], dtype=torch.float32, device="cpu")
        return update