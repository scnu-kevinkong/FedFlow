# -*- coding: utf-8 -*-
import time
from copy import deepcopy
import torch
import numpy as np
import tools # Assuming tools.py is in the path or same directory
from servers.server_base import Server
# Make sure the import path for ClientFedPac is correct
from clients.client_fedpac import ClientFedPac

class ServerFedPac(Server):
    def __init__(self, args):
        super().__init__(args)
        # Ensure ClientFedPac is initialized correctly
        self.clients = [
            ClientFedPac(args, i) for i in range(self.num_clients)
        ]
        self.args = args
        # Use local_epochs from args if available, otherwise default
        self.local_epoch = getattr(args, 'local_epochs', 5) # Name consistency 'local_epochs'
        self.agg_g = getattr(args, 'agg_g', 1) # Option to disable classifier aggregation via args

    def send_models(self):
        """Sends the current global model state (feature extractor part) to active clients."""
        # It might be more efficient to send only the non-classifier weights
        # but sending the whole model is simpler if client handles separation.
        # The client's update_base_model should only update non-classifier parts.
        current_global_state = deepcopy(self.model.state_dict())
        for c in self.active_clients:
            c.set_model(self.model) # Sends the whole model object reference initially
            # Optionally, explicitly update the base model part here if set_model doesn't do it
            # c.update_base_model(current_global_state)

    def train(self):
        for r in range(1, self.global_rounds + 1):
            start_time = time.time()
            # Note: Original code sampled idx_users *after* send_models.
            # Sampling before ensures we only interact with chosen clients.
            self.sample_active_clients() # Determines self.active_clients

            # Get the actual indices of active clients
            active_client_indices = [c.client_idx for c in self.active_clients]
            m = len(active_client_indices)
            if m == 0:
                 print(f"Round [{r}/{self.global_rounds}]\t No clients selected. Skipping round.")
                 continue # Skip round if no clients are active

            # print(f"Round [{r}/{self.global_rounds}]\t Participating clients: {active_client_indices}")

            # Send model state AFTER sampling active clients
            self.send_models() # Sends model to clients in self.active_clients

            local_weights = []      # List to store state_dicts from clients
            local_losses_ce = []    # List for CE loss component
            local_losses_proto = [] # List for proto loss component
            local_accs_rep = []     # List for accuracy from representation phase
            client_agg_weights = [] # Aggregation weights for feature extractor (based on data size)
            client_sizes_label = [] # List of tensors (counts per class) from clients
            client_local_protos = []# List of dictionaries (local protos) from clients

            client_Vars = [] # List for variance V_k from clients
            client_Hs = []   # List for H_k from clients

            # Collect statistics and perform local training on active clients
            for client_idx in active_client_indices:
                local_client = self.clients[client_idx]
                try:
                    # Statistics Collection
                    # Ensure statistics_extraction uses the model state *before* local training
                    v, h = local_client.statistics_extraction()
                    client_Vars.append(v) # Append scalar variance
                    client_Hs.append(h)   # Append tensor H_k [num_classes, d]

                    # Local Training
                    # The client's train method now returns: w, loss_ce, loss_proto, acc_rep, acc_rep, protos
                    w, loss_ce, loss_proto, acc_rep1, acc_rep2, protos = local_client.train()

                    # Store results
                    local_weights.append(deepcopy(w))
                    local_losses_ce.append(loss_ce)
                    local_losses_proto.append(loss_proto)
                    # We only need one accuracy value from the rep phase realistically
                    local_accs_rep.append(acc_rep1) # Using the first returned acc
                    client_agg_weights.append(local_client.agg_weight) # Store data size weight
                    client_sizes_label.append(local_client.sizes_label) # Store class counts tensor
                    client_local_protos.append(deepcopy(protos)) # Store local protos dict

                except Exception as e:
                    print(f"Error training client {client_idx} in round {r}: {e}")
                    # Optionally handle error, e.g., skip client for this round's aggregation

            # Check if any clients successfully trained
            if not local_weights:
                print(f"Round [{r}/{self.global_rounds}]\t No clients completed training. Skipping aggregation.")
                continue

            # --- Aggregation ---

            # 1. Aggregate Feature Extractor (using FedAvg based on data size)
            # Ensure weights are tensors and on the correct device
            agg_weights_tensor = torch.stack(client_agg_weights).to(self.device)
            global_weight_new = tools.average_weights_weighted(local_weights, agg_weights_tensor, exclude_keys=self.clients[0].w_local_keys)
            # Update the server's main model (only non-classifier parts)
            server_state = self.model.state_dict()
            for k in server_state.keys():
                if k not in self.clients[0].w_local_keys:
                    server_state[k] = global_weight_new[k].to(server_state[k].device).to(server_state[k].dtype)
            self.model.load_state_dict(server_state)


            # 2. Aggregate Global Prototypes
            # Ensure sizes_label are tensors
            global_protos = tools.protos_aggregation(client_local_protos, client_sizes_label)


            # --- Distribute Updates ---
            # Send updated feature extractor (implicitly via self.model state) and new global prototypes
            # Original code updated ALL clients here. Sticking to that for now.
            # Consider updating only active clients if that's intended.
            num_all_users = self.num_clients # Total number of clients
            for idx in range(num_all_users):
                 client_to_update = self.clients[idx]
                 # update_base_model takes the aggregated weights
                 client_to_update.update_base_model(global_weight=global_weight_new)
                 # update_global_protos takes the new dictionary of protos
                 client_to_update.update_global_protos(global_protos=global_protos)


            # 3. Aggregate Classifiers (Personalized) - only for active clients
            # Calculate personalized weights alpha_i for each active client i
            if self.agg_g and r < self.global_rounds and client_Vars and client_Hs: # Ensure V and H were collected
                try:
                    # Ensure Vars and Hs are correctly formatted (list of scalars, list of tensors)
                    avg_weights_alphas = tools.get_head_agg_weight(m, client_Vars, client_Hs, device=self.device) # Pass device

                    # Apply personalized aggregation to each active client
                    for i, client_idx in enumerate(active_client_indices):
                        local_client = self.clients[client_idx]
                        alpha_i = avg_weights_alphas[i] # Get the weights for this client

                        if alpha_i is not None:
                            # Aggregate classifiers from *all* participants using weights alpha_i
                            new_cls = tools.agg_classifier_weighted_p(
                                local_weights,         # Weights from all participants this round
                                alpha_i,               # Personal weights for client i
                                local_client.w_local_keys, # Classifier keys
                                i                      # Index of the target client within the *active* list
                            )
                        else:
                            # QP failed for this client, use its own trained classifier
                            print(f"QP solver failed for client {client_idx}, using local classifier.")
                            new_cls = local_weights[i] # Use its own weights from this round

                        # Update the client's local classifier
                        local_client.update_local_classifier(new_weight=new_cls)
                except Exception as e:
                    print(f"Error during classifier aggregation in round {r}: {e}")
                    # If aggregation fails, clients keep their own classifiers from local training

            # --- Logging ---
            loss_avg_ce = sum(local_losses_ce) / len(local_losses_ce) if local_losses_ce else 0
            loss_avg_proto = sum(local_losses_proto) / len(local_losses_proto) if local_losses_proto else 0
            acc_avg_rep = sum(local_accs_rep) / len(local_accs_rep) if local_accs_rep else 0

            train_time = time.time() - start_time

            if r % self.eval_gap == 0 or r == self.global_rounds:
                # Perform evaluation
                # Note: Pass the *current* model state to clients for evaluation
                ptest_acc, ptest_loss, ptest_acc_std = self.evaluate_personalized()
                test_acc, test_loss, test_acc_std = self.evaluate_global() # Use a clearer name

                print(
                    f"Round [{r}/{self.global_rounds}]\t"
                    f"Train Loss (CE|Proto) [{loss_avg_ce:.4f}|{loss_avg_proto:.4f}]\t"
                    f"Train Acc (Rep) [{acc_avg_rep:.2f}]\t"
                    f"Test Loss (Global|Pers) [{test_loss:.4f}|{ptest_loss:.4f}]\t"
                    f"Test Acc (Global|Pers) [{test_acc:.2f}({test_acc_std:.2f})|{ptest_acc:.2f}({ptest_acc_std:.2f})]\t"
                    f"Time [{train_time:.2f}s]"
                )
            else:
                 print(
                    f"Round [{r}/{self.global_rounds}]\t"
                    f"Train Loss (CE|Proto) [{loss_avg_ce:.4f}|{loss_avg_proto:.4f}]\t"
                    f"Train Acc (Rep) [{acc_avg_rep:.2f}]\t"
                    f"Time [{train_time:.2f}s]"
                )

            # Store times, losses, accuracies etc. for later analysis if needed
            # self.history['train_loss_ce'].append(loss_avg_ce)
            # self.history['train_loss_proto'].append(loss_avg_proto)
            # ... etc.
            self.train_times.append(train_time)
            self.round_times.append(time.time() - start_time) # Total round time

    def evaluate_global(self):
        """Evaluates the current global model (feature extractor + last aggregated classifier?) on all clients' test sets."""
        # This evaluates the globally averaged model performance.
        total_samples = 0
        weighted_loss = 0
        weighted_acc = 0
        accs = []

        # Use the server's current model state for evaluation
        global_model_state = deepcopy(self.model.state_dict())

        for c in self.clients:
            # Temporarily set the client's model to the global state for evaluation
            original_state = deepcopy(c.model.state_dict())
            c.model.load_state_dict(global_model_state)

            acc, loss = c.evaluate() # c.evaluate() uses the client's test loader
            if c.num_test > 0: # Avoid division by zero if client has no test data
                 accs.append(torch.tensor(acc, device=self.device)) # Collect accs for std dev
                 weighted_loss += loss * c.num_test # loss should already be avg per sample
                 weighted_acc += acc * c.num_test
                 total_samples += c.num_test

            # Restore client's original model state
            c.model.load_state_dict(original_state)

        if total_samples == 0:
            return 0.0, 0.0, 0.0

        avg_acc = weighted_acc / total_samples
        avg_loss = weighted_loss / total_samples
        std_acc = torch.std(torch.stack(accs)).item() if len(accs) > 1 else 0.0

        return avg_acc, avg_loss, std_acc

    def evaluate_personalized(self):
        """Evaluates each client's personalized model on their test set."""
        # This evaluates the performance after the personalized classifier aggregation step.
        total_samples = 0
        weighted_loss = 0
        weighted_acc = 0
        accs = []

        for c in self.clients:
            # Evaluate the client's current model (which includes the personalized classifier)
            # NO c.train() here - evaluate the model as is after aggregation.
            acc, loss = c.evaluate() # c.evaluate() uses the client's test loader and current model
            if c.num_test > 0:
                 accs.append(torch.tensor(acc, device=self.device))
                 weighted_loss += loss * c.num_test
                 weighted_acc += acc * c.num_test
                 total_samples += c.num_test

        if total_samples == 0:
            return 0.0, 0.0, 0.0

        avg_acc = weighted_acc / total_samples
        avg_loss = weighted_loss / total_samples
        std_acc = torch.std(torch.stack(accs)).item() if len(accs) > 1 else 0.0

        return avg_acc, avg_loss, std_acc