from src.Trainer import Trainer
import warnings
import wandb
import pandas as pd
from simpletransformers.classification import ClassificationModel

class IVESweeper(Trainer):

    def __init__(self, train_df:pd.DataFrame, test_df:pd.DataFrame) -> None:

        self._init_defaults()
        self.wandb_project_name = "xuens/ive-classification"
        super().__init__(train_df, test_df)

    def _init_defaults(self) -> None:

        self.set_sweep_name()
        self.set_sweep_method()
        self.set_metric_goal()

    def set_sweep_name(self, sweep_name="DefaultSweep") -> None:

        self.sweep_name = sweep_name

    def set_train_data_name(self, name:str) -> None:

        self.train_data_name = name

    def set_test_data_name(self, name:str) -> None:

        self.test_data_name = name

    def set_sweep_method(self, method="bayes") -> None:

        self.sweep_method = method

    def set_metric_goal(self, metric="val_acc", goal="maximize") -> None:
        
        self.sweep_metric_dict = {"name" : metric, "goal" : goal}

    def set_sweep_epochs(self, epoch_lst:list) -> None:

        self.sweep_epochs = epoch_lst

    def set_sweep_lr(self, _min:float, _max:float) -> None:
        assert(_min <= _max)
        self.sweep_lr = {"min" : _min, "max" : _max}

    def set_learning_rate(self, lr:float) -> None:

        warnings.warn("Warning : Fixing learning rate in the Sweeper")
        return super().set_learning_rate(lr)

    def set_training_epoch(self, epoch:float) -> None:

        warnings.warn("Warning : Fixing training epoch in the Sweeper")
        return super().set_training_epoch(epoch)

    def create_model_args(self) -> None:
        
        super().create_model_args()
        self.model_args.wandb_project = self.wandb_project_name

    def create_model(self) -> None:
        
        self.model = ClassificationModel(
            self.model_type,
            self.model_name,
            num_labels = self.label_count(),
            args = self.model_args,
            use_cuda=False,
            sweep_config=wandb.config,
        )

    def get_sweep_config(self) -> None:

        return {
            "name" : self.sweep_name,
            "method" : self.sweep_method,
            "metric" : self.sweep_metric_dict,
            "parameters" : {
                "num_train_epochs": {"values": self.sweep_epochs},
                "learning_rate": self.sweep_lr,
            }
        }

    def sweep_check(self) -> None:

        self.model_type
        self.model_name
        self.sweep_epochs
        self.sweep_lr
        self.train_data_name
        self.test_data_name

    def run_sweep(self) -> None:

        self.sweep_check()

        wandb.login()
        sweep_id = wandb.sweep(self.get_sweep_config(), project=self.wandb_project_name)

        def train():

            wandb.init(tags=[self.model_name, self.train_data_name, self.test_data_name])
            self.create_model()
            self.model.train_model(self.train_df, eval_df=self.test_df)
            self.model.eval_model(self.test_df)
        
        wandb.agent(sweep_id, train)