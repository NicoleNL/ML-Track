import pandas as pd
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import torch

class Trainer():

    def __init__(self, train_df:pd.DataFrame, test_df:pd.DataFrame) -> None:
        
        self.train_df = train_df
        self.test_df = test_df
        self.set_label_name()
        self.set_text_name()
        self.create_model_args()
        
    def set_label_name(self, label="label") -> None:

        self.label = label

    def set_text_name(self, _text="text") -> None:

        self.text = _text

    def _get_labels(self) -> None:

        return list(set(self.train_df[self.label].tolist()))

    def label_count(self) -> None:

        return len(self._get_labels())
    
    def create_model_args(self) -> None:

        self.model_args = ClassificationArgs()
        self.model_args.reprocess_input_data = True
        self.model_args.overwrite_output_dir = True
        self.model_args.evaluate_during_training = True
        self.model_args.manual_seed = 8
        self.model_args.use_multiprocessing = True
        self.model_args.train_batch_size = 16
        self.model_args.eval_batch_size = 8
        self.model_args.labels_list = self._get_labels()
        
    def set_learning_rate(self, lr:float) -> None:

        self.model_args.learning_rate = lr

    def set_training_epoch(self, epoch:float) -> None:

        self.model_args.num_train_epochs = epoch

    def set_model_type(self, modeltype:str) -> None:

        self.model_type = modeltype

    def set_model_name(self, modelname:str) -> None:

        self.model_name = modelname

    def create_model(self) -> None:

        self.model = ClassificationModel(
            self.model_type,
            self.model_name,
            num_labels = self.label_count(),
            args = self.model_args,
            use_cuda =  torch.cuda.is_available()
        )

    def train_model(self, output_dir="default_output") -> None:

        self.model.train_model(self.train_df, output_dir=output_dir, eval_df=self.test_df)

    def accuracy(self) -> None:

        _, _, wrong_predictions = self.model.eval_model(self.test_df)
        return 1.0 - (len(wrong_predictions)/len(self.test_df))

    def get_predictions(self) -> None:

        predictions, _ = self.model.predict(self.test_df[self.text].tolist())
        
        tmp_df = self.test_df.copy()
        tmp_df["prediction"] = predictions
        
        return tmp_df