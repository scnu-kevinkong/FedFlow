import torch
from copy import deepcopy
from clients.client_base import Client # Assuming client_base defines base Client class
from utils.util import AverageMeter    # Assuming util defines AverageMeter
import traceback # Import traceback for error handling in get_update

class ClientFedAvg(Client):
    def __init__(self, args, client_idx, is_corrupted):
        super().__init__(args, client_idx, is_corrupted)
        # Ensure required attributes are set, e.g., from args or base class
        self.lr = getattr(args, 'lr', 0.01)
        self.momentum = getattr(args, 'momentum', 0.9)
        self.wd = getattr(args, 'wd', 5e-4)
        self.local_epochs = getattr(args, 'local_epochs', 5)
        # self.model is initialized in the base Client class
        # self.loss (criterion) should be defined (e.g., nn.CrossEntropyLoss())
        if not hasattr(self, 'loss') or self.loss is None:
             # Use a default loss if not provided by base class or args
             self.loss = torch.nn.CrossEntropyLoss()
        if not hasattr(self, 'device'):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def train(self, num_steps=None): # Allow specifying num_steps for fine-tuning
        """
        Performs local training on the client's data.

        Args:
            num_steps (int, optional): Number of optimization steps to perform.
                                      If None, runs for self.local_epochs.

        Returns:
            tuple: average training loss (float), statistics dictionary (dict).
                   The dictionary contains {'samples': num_samples_trained}.
        """
        trainloader = self.load_train_data()
        # Handle cases where trainloader might be None or empty
        if trainloader is None:
            print(f"Warning: Client {self.client_idx} could not load train data.")
            return 0.0, {'samples': 0}
        try:
            # Check if dataset is empty before creating iterator
            if len(trainloader.dataset) == 0:
                 print(f"Warning: Client {self.client_idx} has an empty training dataset.")
                 return 0.0, {'samples': 0}
        except Exception:
             # Fallback if len(dataloader.dataset) is not supported
             pass


        # Ensure model is on the correct device before creating optimizer
        self.model = self.model.to(self.device)
        # Filter parameters that require gradients for the optimizer
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                    lr=self.lr, momentum=self.momentum, weight_decay=self.wd)
        self.model.train() # Set model to training mode

        losses = AverageMeter()
        accs = AverageMeter() # Accuracy calculation is optional here

        steps_done = 0
        epochs_to_run = self.local_epochs if num_steps is None else 1 # Run 1 epoch if num_steps is set

        # Check if trainloader actually has data
        try:
            has_data = False
            for _ in trainloader:
                has_data = True
                break
            if not has_data:
                print(f"Warning: Client {self.client_idx}'s trainloader is empty.")
                self.model = self.model.to("cpu")
                return 0.0, {'samples': 0}
        except Exception as e:
             print(f"Error checking/iterating trainloader for client {self.client_idx}: {e}")
             self.model = self.model.to("cpu")
             return 0.0, {'samples': 0}


        # --- Training Loop ---
        for e in range(epochs_to_run):
            for i, batch_data in enumerate(trainloader):
                # Basic input validation
                try:
                    if not isinstance(batch_data, (list, tuple)) or len(batch_data) < 2:
                         print(f"Warning: Client {self.client_idx} received malformed batch data (type: {type(batch_data)}). Skipping batch.")
                         continue
                    x, y = batch_data[0], batch_data[1]
                    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
                        print(f"Warning: Client {self.client_idx} received non-tensor batch data. Skipping batch.")
                        continue
                    x, y = x.to(self.device), y.to(self.device)
                except Exception as data_err:
                     print(f"Error processing batch data for client {self.client_idx}: {data_err}. Skipping batch.")
                     continue

                # Forward pass
                output = self.model(x)
                loss = self.loss(output, y) # self.loss is the criterion

                # Check for NaN/Inf loss
                if torch.isnan(loss) or torch.isinf(loss):
                     print(f"Warning: Client {self.client_idx} encountered NaN/Inf loss. Skipping step.")
                     optimizer.zero_grad() # Still zero grad before next step
                     continue

                # Calculate accuracy (optional)
                acc = (output.argmax(1) == y).float().mean().item() * 100.0
                accs.update(acc, x.size(0))
                losses.update(loss.item(), x.size(0)) # Store loss and sample count

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                # Optional: Gradient Clipping
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                steps_done += 1
                # Stop early if num_steps is specified and reached
                if num_steps is not None and steps_done >= num_steps:
                    break
            # Exit epoch loop if steps are completed
            if num_steps is not None and steps_done >= num_steps:
                    break
        # --- End Training Loop ---

        # Move model back to CPU after training
        self.model = self.model.to("cpu")

        # --- *** CORRECTED RETURN VALUE *** ---
        # Create statistics dictionary
        # Ensure losses.count reflects the actual number of samples processed
        stats = {'samples': losses.count if losses.count > 0 else 0}

        # Return loss average and the stats dictionary
        # Return 0.0 loss if no samples were processed
        final_loss_avg = losses.avg if losses.count > 0 else 0.0
        final_acc_avg = accs.avg if accs.count > 0 else 0.0
        return final_acc_avg, final_loss_avg, stats
        # -------------------------------------


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
