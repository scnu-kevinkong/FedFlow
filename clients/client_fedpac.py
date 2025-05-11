# -*- coding: utf-8 -*-
from utils.util import AverageMeter
import torch
from clients.client_base import Client
import copy
import tools

class ClientFedPac(Client):
    def __init__(self, args, client_idx):
        super().__init__(args, client_idx)
        self.args = args
        self.num_classes = args.num_classes
        # Ensure model has the classifier_weight_keys attribute
        if not hasattr(self.model, 'classifier_weight_keys'):
             raise AttributeError("Model definition must include 'classifier_weight_keys' attribute.")
        if not hasattr(self.model, 'D'):
             raise AttributeError("Model definition must include 'D' attribute for feature dimension.")
        self.w_local_keys = self.model.classifier_weight_keys
        self.feature_dim = self.model.D # Store feature dimension
        # Setting local_ep_rep directly here might override args? Consider using args.local_epochs
        self.local_ep_rep = getattr(args, 'local_ep_rep', 1) # Number of epochs for representation learning
        self.epoch_classifier = getattr(args, 'epoch_classifier', 1) # Number of epochs for classifier learning

        self.probs_label = self.prior_label(self.load_train_data()).to(self.device)
        self.sizes_label = self.size_label(self.load_train_data()).to(self.device)
        self.datasize = torch.tensor(len(self.load_train_data().dataset), dtype=torch.float, device=self.device) # Use float for potential division
        self.agg_weight = self.aggregate_weight()
        self.global_protos = {} # Will store {class_idx: proto_tensor}
        # Removed self.g_protos and self.g_classes as they seemed unused after update_global_protos
        self.mse_loss = torch.nn.MSELoss()
        self.lam = getattr(args, 'lam', 1.0) # Lambda for proto loss, get from args or default to 1.0

    def prior_label(self, dataset):
        py = torch.zeros(self.num_classes, device=self.device) # Move to device early
        total = len(dataset.dataset)
        # Use DataLoader directly for efficiency
        data_loader = torch.utils.data.DataLoader(dataset.dataset, batch_size=self.batch_size, shuffle=False)
        for _, labels in data_loader:
            labels = labels.to(self.device)
            for i in range(self.num_classes):
                py[i] += (labels == i).sum()
        # Handle case where total is 0
        if total > 0:
            py = py / total
        else:
            py.fill_(0.0) # Or 1.0/self.num_classes if uniform prior is desired for empty datasets
        return py

    def size_label(self, dataset):
        # size_label is just prior_label * total_samples
        # We already calculate prior_label, let's reuse it if possible
        # Re-calculating for simplicity here, but consider optimization
        ps = torch.zeros(self.num_classes, device=self.device) # Move to device early
        total = len(dataset.dataset)
        data_loader = torch.utils.data.DataLoader(dataset.dataset, batch_size=self.batch_size, shuffle=False)
        for _, labels in data_loader:
            labels = labels.to(self.device)
            for i in range(self.num_classes):
                ps[i] += (labels == i).sum()
        # No division needed here, these are counts
        return ps # Returns tensor of counts per class

    def aggregate_weight(self):
        # Weight based on data size
        data_size = len(self.load_train_data().dataset)
        w = torch.tensor(data_size, dtype=torch.float, device=self.device) # Use float
        return w

    def local_test(self, test_loader):
        model = self.model
        model.eval()
        model.to(self.device) # Ensure model is on device for testing
        correct = 0
        total = 0 # Initialize total
        loss_test = 0.0 # Initialize loss
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # Ensure model returns only outputs when return_feat=False
                outputs = model(inputs, return_feat=False)
                loss = self.loss(outputs, labels)
                loss_test += loss.item() * inputs.size(0) # Weighted loss sum
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0) # Increment total samples processed

        model.to("cpu") # Move back to CPU if needed elsewhere
        if total == 0:
             return 0.0, 0.0 # Handle empty test loader case
        acc = 100.0 * correct / total
        avg_loss = loss_test / total
        return acc, avg_loss

    def update_base_model(self, global_weight):
        local_weight = self.model.state_dict()
        w_local_keys = self.w_local_keys
        for k in local_weight.keys():
            if k not in w_local_keys:
                # Ensure the dtype and device match
                local_weight[k] = global_weight[k].to(local_weight[k].device).to(local_weight[k].dtype)
        self.model.load_state_dict(local_weight)

    def update_local_classifier(self, new_weight):
        local_weight = self.model.state_dict()
        w_local_keys = self.w_local_keys
        for k in local_weight.keys():
            if k in w_local_keys:
                 # Ensure the dtype and device match
                local_weight[k] = new_weight[k].to(local_weight[k].device).to(local_weight[k].dtype)
        self.model.load_state_dict(local_weight)

    def update_global_protos(self, global_protos):
        # Store the received global prototypes
        # Ensure protos are on the correct device
        self.global_protos = {k: v.to(self.device) for k, v in global_protos.items()}
        # No need to create g_classes, g_protos here as self.global_protos holds the data

    def get_local_protos(self):
        model = self.model
        model.eval() # Ensure model is in eval mode for consistent features
        model.to(self.device)
        local_protos_list = {} # {class_idx: [list of feature tensors]}
        trainloader = self.load_train_data()

        with torch.no_grad():
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                features, _ = model(inputs, return_feat=True) # Get features
                # No need to clone().detach() features if using torch.no_grad()
                for i in range(len(labels)):
                    label_item = labels[i].item()
                    if label_item not in local_protos_list:
                        local_protos_list[label_item] = []
                    local_protos_list[label_item].append(features[i]) # Append the feature tensor directly

        # Calculate mean proto per class
        local_protos_mean = {}
        for label, protos in local_protos_list.items():
            if protos: # Ensure list is not empty
                local_protos_mean[label] = torch.stack(protos).mean(dim=0)

        model.to("cpu") # Move back to CPU if needed
        return local_protos_mean # Return dictionary {class_idx: mean_proto_tensor}

    def statistics_extraction(self):
        model = self.model
        model.eval()
        model.to(self.device)

        # Use stored feature dimension
        d = self.feature_dim

        feature_dict = {} # {class_idx: [list of feature tensors]}
        trainloader = self.load_train_data()
        with torch.no_grad():
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                features, _ = model(inputs, return_feat=True)
                for i in range(len(labels)):
                    yi = labels[i].item()
                    if yi not in feature_dict:
                        feature_dict[yi] = []
                    feature_dict[yi].append(features[i]) # No detach needed

        # Stack features for easier computation
        stacked_feature_dict = {}
        for k, v in feature_dict.items():
            if v: # Ensure list is not empty
                stacked_feature_dict[k] = torch.stack(v) # Shape: [num_samples_k, d]

        # Use pre-calculated priors and ensure it's on the correct device
        py = self.probs_label.to(self.device) # Proportions per class
        py2 = py.mul(py)

        v = 0.0 # Variance term V_k
        h_ref = torch.zeros((self.num_classes, d), device=self.device) # H_k term

        for k in range(self.num_classes):
            if k in stacked_feature_dict:
                feat_k = stacked_feature_dict[k] # Features for class k [num_k, d]
                num_k = feat_k.shape[0]
                if num_k > 0: # Avoid division by zero
                    feat_k_mu = feat_k.mean(dim=0) # Mean feature for class k [d]
                    h_ref[k] = py[k] * feat_k_mu # Weighted mean feature

                    # Variance calculation: E[||x||^2] - ||E[x]||^2
                    # More stable variance calc: mean(sum(x_i^2)) - sum(mean(x_i)^2) per dim?
                    # Original paper's formula: (py[k] * trace(feat_k^T @ feat_k / num_k)) - py2[k] * ||feat_k_mu||^2
                    # Let's stick to the presumed formula from the code provided
                    term1 = py[k] * torch.trace(torch.mm(feat_k.t(), feat_k) / num_k)
                    term2 = py2[k] * torch.sum(feat_k_mu * feat_k_mu)
                    v += (term1 - term2).item()

        # Ensure model is back on CPU if needed
        model.to("cpu")

        # Normalize variance by total data size if required by algorithm definition
        # The original code divides by self.datasize.item() here. Check if FedPAC paper does this.
        # Assuming it does:
        if self.datasize.item() > 0:
            v = v / self.datasize.item()
        else:
            v = 0.0

        return v, h_ref # Return scalar variance, tensor H_k [num_classes, d]

    def train(self):
        trainloader = self.load_train_data()

        # Use configured epochs
        epoch_classifier = self.epoch_classifier
        local_ep_rep = self.local_ep_rep

        # Average meters for tracking losses and accuracy
        losses_cls = AverageMeter()
        accs_cls = AverageMeter()
        losses_rep = AverageMeter()
        accs_rep = AverageMeter()
        avg_loss_ce = AverageMeter() # For cross-entropy loss component
        avg_loss_proto = AverageMeter() # For proto loss component

        # --- Phase 1: Train Classifier ---
        self.model.train() # Set model to train mode
        # Freeze representation layers, unfreeze classifier layers
        for name, param in self.model.named_parameters():
            param.requires_grad = (name in self.w_local_keys)

        # Use separate optimizer for classifier? Original code suggests so.
        # Consider using args.lr_g or a specific classifier LR
        lr_g = getattr(self.args, 'lr_g', 0.1) # Classifier learning rate
        optimizer_cls = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr_g,
                                    momentum=self.momentum, weight_decay=self.wd) # Use hyperparams from args

        for ep in range(epoch_classifier):
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)
                self.model.to(self.device)
                # Get output only, no need for features here
                output = self.model(x, return_feat=False)
                loss = self.loss(output, y)

                optimizer_cls.zero_grad()
                loss.backward()
                optimizer_cls.step()

                # Track metrics
                acc = (output.argmax(1) == y).float().mean() * 100.0
                accs_cls.update(acc.item(), x.size(0))
                losses_cls.update(loss.item(), x.size(0))

        # --- Phase 2: Train Representation ---
        self.model.train() # Keep model in train mode
        # Unfreeze representation layers, freeze classifier layers
        for name, param in self.model.named_parameters():
            param.requires_grad = not (name in self.w_local_keys)

        # Use main optimizer for representation
        optimizer_rep = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr,
                                    momentum=self.momentum, weight_decay=self.wd) # Use hyperparams from args

        # Get local prototypes *before* representation training starts for this round
        # These serve as stable targets if global protos are missing
        local_protos_stable = self.get_local_protos() # {label: proto_tensor}

        for ep in range(local_ep_rep):
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)
                self.model.to(self.device)
                optimizer_rep.zero_grad()

                # Get both features (protos) and output
                protos, output = self.model(x, return_feat=True)

                # Loss 0: Standard Cross-Entropy
                loss0 = self.loss(output, y)

                # Loss 1: Proto Regularization
                loss1_val = 0.0 # Use a float for accumulation
                target_protos = torch.zeros_like(protos) # Initialize target protos tensor
                valid_proto_indices = [] # Track indices with valid targets

                if self.global_protos or local_protos_stable: # Only compute loss if protos exist
                    for idx_p in range(len(y)):
                        yi = y[idx_p].item()
                        found_proto = False
                        if yi in self.global_protos:
                            target_protos[idx_p] = self.global_protos[yi] # No detach needed if optimizer ignores classifier
                            found_proto = True
                        elif yi in local_protos_stable:
                            target_protos[idx_p] = local_protos_stable[yi] # Use stable local proto
                            found_proto = True
                        # else: target_protos[idx_p] remains zero

                        if found_proto:
                             valid_proto_indices.append(idx_p)

                    # Calculate MSE only for samples with valid target protos
                    if valid_proto_indices:
                         valid_indices_tensor = torch.tensor(valid_proto_indices, device=self.device)
                         # Select the corresponding rows from protos and target_protos
                         protos_for_loss = protos.index_select(0, valid_indices_tensor)
                         target_protos_for_loss = target_protos.index_select(0, valid_indices_tensor)
                         loss1 = self.mse_loss(protos_for_loss, target_protos_for_loss)
                         loss1_val = loss1.item() # Store scalar value for tracking
                    else:
                         loss1 = 0.0 # No valid protos, MSE loss is 0

                else:
                    loss1 = 0.0 # No protos available at all

                # Total Loss
                loss = loss0 + self.lam * loss1

                loss.backward()
                optimizer_rep.step()

                # Track metrics
                acc = (output.argmax(1) == y).float().mean() * 100.0
                accs_rep.update(acc.item(), x.size(0))
                losses_rep.update(loss.item(), x.size(0))
                avg_loss_ce.update(loss0.item(), x.size(0))
                # Use the scalar loss1_val for tracking proto loss average
                avg_loss_proto.update(loss1_val, x.size(0))

        # Get final state dict and computed protos for this round
        w = self.model.state_dict()
        # Get the latest local prototypes after training
        current_local_protos = self.get_local_protos()

        # Move model back to CPU before returning weights
        self.model.to("cpu")

        # Return values:
        # w: model state dictionary
        # avg_loss_ce.avg: Average Cross-Entropy loss during representation training
        # avg_loss_proto.avg: Average Proto MSE loss during representation training
        # accs_rep.avg: Average accuracy during representation training (used twice as placeholder)
        # current_local_protos: Dictionary of calculated local prototypes for this client
        return w, avg_loss_ce.avg, avg_loss_proto.avg, accs_rep.avg, accs_rep.avg, current_local_protos