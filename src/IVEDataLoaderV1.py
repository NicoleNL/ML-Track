import pandas as pd
from src.IVEDataLoaderBase import IVEDataLoaderBase

class IVEDataLoaderV1(IVEDataLoaderBase):

    def __init__(self, raw_ive_df:pd.DataFrame, target="component_affected", label="label") -> None: 
        """Version 1: Load training and test data"""
        super().__init__(raw_ive_df)
        self.initialize_data()
        self.target = target
        self.label = label

    def generate_train_test(self, test_size=10, label_dct=None) -> None:
        """Run train and test data generations

        Args:
            test_size (int, optional): _description_. Defaults to 10.
            label_dct (_type_, optional): _description_. Defaults to None.
        """       
        #create text label of main category and convert to numeric
        self.set_conversion_dict(label_dct)
        self._create_labels()
        
        #split data into training and test
        self.set_test_size(test_size)
        self._train_test_split()
        self._shuffle_train_df()
        self.format_train_test_df()

    def _create_labels(self) -> None:
        """Create text labels of main categories
        """
        raw_lst = self.all_df[self.target].tolist()
        self.all_df[self.label] = [x.split(",")[0].split(".")[0] for x in raw_lst]
        self.set_labels_remove_lst()
        self.filter_labels()
        self._labels_to_numeric()

    def _labels_to_numeric(self):
        """Converts labels to numetic form based on existing dictionary
        """
        label_lst = self.all_df[self.label].tolist()
        label_lst = [self.label_dct[x] for x in label_lst]

        self.all_df[self.label] = label_lst

    def set_labels_remove_lst(self, lst=['doc', 'soc']):
        """Set list of labels where the sample size is too small

        Args:
            lst (list, optional): list of labels(str) to remove. Defaults to ['doc', 'soc'].
        """
        self.remove_lst = lst

    def format_train_test_df(self):
        """Format train and test data for ML training runs
        """
        target = ["title", "label"]
        self.train_df = self.train_df[target] 
        self.test_df = self.test_df[target]

    def filter_labels(self):
        """Remove labels with sample size that is too small
        """
        self.all_df = self.all_df[~self.all_df[self.label].isin(self.remove_lst)]

    def set_test_size(self, size=10) -> None:
        """Set test size for each label/category

        Args:
            size (int, optional): size of test data for each label. Defaults to 10.
        """
        self.test_size = size

    def _train_test_split(self) -> None:
        """Split whole dataset into training set and test set
        """
        train_lst = []
        eval_lst = []
        grouped = self.all_df.groupby(self.label)

        for _, label_df in grouped:
            train_lst.append(label_df[self.test_size:])
            eval_lst.append(label_df[:self.test_size])

        self.train_df = pd.concat(train_lst)
        self.test_df = pd.concat(eval_lst)

    def set_conversion_dict(self, dct=None) -> None:
        """Set dictionary to convert text labels to numeric

        Args:
            dct (_type_, optional): Conversion dictionary. Defaults to None.
        """
        if dct is None:
            dct = {
                "bios" : 0,
                "board" : 1,
                "fw" : 2,
                "hw" : 3,
                "ip" : 4,
                "sw" : 5,
                "val" : 6,
                "other" : 7
            }

        self.label_dct = dct

    def _shuffle_train_df(self) -> None:
        """Shuffle training data
        """
        self.train_df = self.train_df.sample(frac=1)

    def save_train_df(self, path="train-test-data/train_V1.csv") -> None:
        """Save training dataframe to specific path

        Args:
            path (str, optional): path to save training df. Defaults to "train-test-data/train_V1.csv".
        """
        self.train_df.to_csv(path) 

    def save_test_df(self, path="train-test-data/test_V1.csv") -> None:
        """Save test dataframe to specific path

        Args:
            path (str, optional): path to save test df. Defaults to "train-test-data/test_V1.csv".
        """
        self.test_df.to_csv(path) 