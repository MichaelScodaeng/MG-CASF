import os
import torch
import torch.nn as nn
import logging
# Import MemoryModel to use for isinstance checks
# This import path assumes MemoryModel is in models/MemoryModel.py relative to your project root.
# If this path is incorrect, you might need to adjust it or remove the isinstance checks
# and rely purely on hasattr, though isinstance is generally safer for type checking.
try:
    from models.MemoryModel import MemoryModel
except ImportError:
    logging.warning("Could not import models.MemoryModel. `isinstance` checks for MemoryModel will be skipped in EarlyStopping.")
    class MemoryModel: # Define a dummy class if import fails to prevent NameError
        pass


class EarlyStopping(object):

    def __init__(self, patience: int, save_model_folder: str, save_model_name: str, logger: logging.Logger, model_name: str = None):
        """
        Early stop strategy.
        :param patience: int, max patience
        :param save_model_folder: str, save model folder
        :param save_model_name: str, save model name
        :param logger: Logger
        :param model_name: str, model name
        """
        self.patience = patience
        self.counter = 0
        self.best_metrics = {}
        self.early_stop = False
        self.logger = logger
        self.save_model_path = os.path.join(save_model_folder, f"{save_model_name}.pkl")
        self.model_name = model_name
        if self.model_name in ['JODIE', 'DyRep', 'TGN']:
            # path to additionally save the nonparametric data (e.g., tensors) in memory-based models (e.g., JODIE, DyRep, TGN)
            self.save_model_nonparametric_data_path = os.path.join(save_model_folder, f"{save_model_name}_nonparametric_data.pkl")

    def step(self, metrics: list, model: nn.Module):
        """
        execute the early stop strategy for each evaluation process
        :param metrics: list, list of metrics, each element is a tuple (str, float, boolean) -> (metric_name, metric_value, whether higher means better)
        :param model: nn.Module
        :return:
        """
        metrics_compare_results = []
        for metric_tuple in metrics:
            metric_name, metric_value, higher_better = metric_tuple[0], metric_tuple[1], metric_tuple[2]

            if higher_better:
                if self.best_metrics.get(metric_name) is None or metric_value >= self.best_metrics.get(metric_name):
                    metrics_compare_results.append(True)
                else:
                    metrics_compare_results.append(False)
            else:
                if self.best_metrics.get(metric_name) is None or metric_value <= self.best_metrics.get(metric_name):
                    metrics_compare_results.append(True)
                else:
                    metrics_compare_results.append(False)
        # all the computed metrics are better than the best metrics
        if torch.all(torch.tensor(metrics_compare_results)):
            for metric_tuple in metrics:
                metric_name, metric_value = metric_tuple[0], metric_tuple[1]
                self.best_metrics[metric_name] = metric_value
            self.save_checkpoint(model)
            self.counter = 0
        # metrics are not better at the epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def save_checkpoint(self, model: nn.Module):
        """
        saves model at self.save_model_path
        :param model: nn.Module
        :return:
        """
        self.logger.info(f"save model {self.save_model_path}")
        torch.save(model.state_dict(), self.save_model_path)

        # This block is only executed for memory-based models ('JODIE', 'DyRep', 'TGN')
        if self.model_name in ['JODIE', 'DyRep', 'TGN']:
            # model[0] is the backbone, which could be a MemoryModel directly or a CCASFWrapper wrapping a MemoryModel
            backbone_module = model[0] 

            memory_model_instance = None
            if hasattr(backbone_module, 'backbone_model') and isinstance(backbone_module.backbone_model, MemoryModel):
                # This case handles when the MemoryModel (TGN, DyRep, JODIE) is wrapped by CCASFWrapper
                memory_model_instance = backbone_module.backbone_model
            elif isinstance(backbone_module, MemoryModel):
                # This case handles when the MemoryModel is the direct backbone (not wrapped by CCASFWrapper)
                memory_model_instance = backbone_module
            
            if memory_model_instance is not None and hasattr(memory_model_instance, 'memory_bank'):
                torch.save(memory_model_instance.memory_bank.node_raw_messages, self.save_model_nonparametric_data_path)
                self.logger.info(f"Non-parametric data (memory_bank.node_raw_messages) saved to {self.save_model_nonparametric_data_path}")
            else:
                self.logger.warning(f"Model '{self.model_name}' is a memory-based model, but 'memory_bank' could not be found "
                                    f"in {type(backbone_module).__name__} or its 'backbone_model' attribute. "
                                    "Non-parametric data not saved.")


    def load_checkpoint(self, model: nn.Module, map_location: str = None):
        """
        load model at self.save_model_path
        :param model: nn.Module
        :param map_location: str, how to remap the storage locations
        :return:
        """
        self.logger.info(f"load model {self.save_model_path}")
        model.load_state_dict(torch.load(self.save_model_path, map_location=map_location))
        
        # This block is only executed for memory-based models ('JODIE', 'DyRep', 'TGN')
        if self.model_name in ['JODIE', 'DyRep', 'TGN']:
            backbone_module = model[0]
            memory_model_instance = None
            if hasattr(backbone_module, 'backbone_model') and isinstance(backbone_module.backbone_model, MemoryModel):
                memory_model_instance = backbone_module.backbone_model
            elif isinstance(backbone_module, MemoryModel):
                memory_model_instance = backbone_module

            if memory_model_instance is not None and hasattr(memory_model_instance, 'memory_bank'):
                # Load the raw messages directly into the memory bank
                memory_model_instance.memory_bank.node_raw_messages = torch.load(self.save_model_nonparametric_data_path, map_location=map_location)
                self.logger.info(f"Non-parametric data (memory_bank.node_raw_messages) loaded from {self.save_model_nonparametric_data_path}")
            else:
                self.logger.warning(f"Model '{self.model_name}' is a memory-based model, but 'memory_bank' could not be found "
                                    f"in {type(backbone_module).__name__} or its 'backbone_model' attribute. "
                                    "Non-parametric data not loaded.")