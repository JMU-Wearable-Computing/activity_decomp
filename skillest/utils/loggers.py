import torch
import wandb


class SubjectMovementLogger():
    def __init__(self, num_sub=10, num_mov=10, 
                 prefix=None, 
                 cols=list(range(1, 11))):

        self.prefix = prefix
        self.num_sub = num_sub
        self.num_mov = num_mov
        self.cols = cols
        self.accuracies = {} 
        self.subject = None
        self.movement = None
    
    def set_subject_movement(self, subject, movement):
        self.subject = subject
        self.movement = movement
    
    def log(self, acc, mode="train"):
        if mode not in self.accuracies.keys():
            self.accuracies[mode] = torch.zeros([self.num_sub, self.num_mov])

        self.accuracies[mode][self.subject - 1, self.movement - 1] += acc
    
    def reduce(self):
        prefix = f"{self.prefix}_" if self.prefix else ""
        for mode, accuracies in self.accuracies.items():
            subject_table = wandb.Table(columns=self.cols, data=[accuracies.mean(axis=0).tolist()])
            movement_table = wandb.Table(columns=self.cols, data=[accuracies.mean(axis=1).tolist()])
            accuracy_table = wandb.Table(columns=self.cols, data=accuracies.tolist())
            wandb.log({f"{prefix}{mode}_subject_accuracies": subject_table, 
                    f"{prefix}{mode}_movement_accuracies": movement_table, 
                    f"{prefix}{mode}_accuracies": accuracy_table})
