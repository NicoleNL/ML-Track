from abc import ABC,abstractmethod
import pandas as pd

class IVEDataLoaderBase(ABC):

    def __init__(self, raw_ive_df:pd.DataFrame) -> None:
        """Base class to load ML train/test data for IVE

        Args:
            raw_ive_df (pd.DataFrame): raw pandas dataframe containing IVE data 
        """
        self.raw_df = raw_ive_df

    def initialize_data(self, cols=["id", "title", "component_affected"]) -> None:
        """Initialize data contain only relevant subset
        """

        self.select_columns(cols)
        self.all_df = self.raw_df[self.cols]

    def select_columns(self, cols=["id", "title", "component_affected"]) -> None:
        """Subset of data that is relevant

        Args:
            cols (list, optional): list of relevant columns. Defaults to ["id", "title", "component_affected"].
        """
        self.cols = cols
        
    @abstractmethod
    def _create_labels(self):
        """Create labels for training purposes
        """
        pass

    @abstractmethod
    def _train_test_split(self):
        """Split data into train test portions
        """
        pass