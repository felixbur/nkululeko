# opensmileset.py
"""Module for extracting OpenSMILE features from audio files.
OpenSMILE is an audio feature extraction toolkit supporting various feature sets.
"""
import os
import logging
from typing import Optional, Union, List, Any, Dict

import opensmile
import pandas as pd
import numpy as np

import nkululeko.glob_conf as glob_conf
from nkululeko.feat_extract.featureset import Featureset


class Opensmileset(Featureset):
    """Class for extracting OpenSMILE features from audio files.

    This class provides methods to extract various OpenSMILE feature sets like eGeMAPSv02,
    ComParE_2016, etc. at different feature levels (LowLevelDescriptors or Functionals).

    Attributes:
        featset (str): The OpenSMILE feature set to extract (e.g., 'eGeMAPSv02')
        feature_set: The OpenSMILE feature set object
        featlevel (str): The feature level ('LowLevelDescriptors' or 'Functionals')
        feature_level: The OpenSMILE feature level object
    """

    # Available feature sets for validation
    AVAILABLE_FEATURE_SETS = ["eGeMAPSv02", "ComParE_2016", "GeMAPSv01a", "eGeMAPSv01a"]

    # Available feature levels for validation
    AVAILABLE_FEATURE_LEVELS = ["LowLevelDescriptors", "Functionals"]

    def __init__(
        self,
        name: str,
        data_df: pd.DataFrame,
        feats_type: Optional[str] = None,
        config_file: Optional[str] = None,
    ):
        """Initialize the Opensmileset class.

        Args:
            name (str): Name of the feature set
            data_df (pd.DataFrame): DataFrame containing audio file paths
            feats_type (Optional[str]): Type of features to extract
            config_file (Optional[str]): Configuration file path
        """
        super().__init__(name, data_df, feats_type)

        # Get feature set configuration
        self.featset = self.util.config_val("FEATS", "set", "eGeMAPSv02")

        # Validate and set feature set
        if self.featset not in self.AVAILABLE_FEATURE_SETS:
            self.util.warning(
                f"Feature set '{self.featset}' might not be supported. "
                f"Available sets: {', '.join(self.AVAILABLE_FEATURE_SETS)}"
            )

        try:
            self.feature_set = eval(f"opensmile.FeatureSet.{self.featset}")
        except (AttributeError, SyntaxError) as e:
            self.util.error(f"Invalid feature set: {self.featset}. Error: {str(e)}")
            raise ValueError(f"Invalid feature set: {self.featset}")

        # Get feature level configuration
        self.featlevel = self.util.config_val("FEATS", "level", "functionals")

        # Convert shorthand names to full OpenSMILE names
        if self.featlevel == "lld":
            self.featlevel = "LowLevelDescriptors"
        elif self.featlevel == "functionals":
            self.featlevel = "Functionals"

        # Validate and set feature level
        if self.featlevel not in self.AVAILABLE_FEATURE_LEVELS:
            self.util.warning(
                f"Feature level '{self.featlevel}' might not be supported. "
                f"Available levels: {', '.join(self.AVAILABLE_FEATURE_LEVELS)}"
            )

        try:
            self.feature_level = eval(f"opensmile.FeatureLevel.{self.featlevel}")
        except (AttributeError, SyntaxError) as e:
            self.util.error(f"Invalid feature level: {self.featlevel}. Error: {str(e)}")
            raise ValueError(f"Invalid feature level: {self.featlevel}")

    def extract(self) -> pd.DataFrame:
        """Extract the features based on the initialized dataset or load them from disk if available.

        This method checks if features are already stored on disk and loads them if available,
        otherwise it extracts features using OpenSMILE.

        Returns:
            pd.DataFrame: DataFrame containing the extracted features

        Raises:
            RuntimeError: If feature extraction fails
        """
        store = self.util.get_path("store")
        store_format = self.util.config_val("FEATS", "store_format", "pkl")
        storage = f"{store}{self.name}.{store_format}"

        # Check if we need to extract features or use existing ones
        extract = eval(
            self.util.config_val("FEATS", "needs_feature_extraction", "False")
        )
        no_reuse = eval(self.util.config_val("FEATS", "no_reuse", "False"))

        if extract or not os.path.isfile(storage) or no_reuse:
            self.util.debug("Extracting OpenSMILE features, this might take a while...")

            try:
                smile = opensmile.Smile(
                    feature_set=self.feature_set,
                    feature_level=self.feature_level,
                    num_workers=self.n_jobs,
                    verbose=True,
                )

                # Extract features based on index type
                if isinstance(self.data_df.index, pd.MultiIndex):
                    self.df = smile.process_index(self.data_df.index)
                    self.df = self.df.set_index(self.data_df.index)
                else:
                    self.df = smile.process_files(self.data_df.index)
                    # Clean up the index
                    if self.df.index.nlevels > 1:
                        self.df.index = self.df.index.droplevel(1)
                        self.df.index = self.df.index.droplevel(1)

                # Save extracted features
                self.util.write_store(self.df, storage, store_format)

                # Update configuration to avoid re-extraction
                try:
                    glob_conf.config["DATA"]["needs_feature_extraction"] = "False"
                except KeyError:
                    pass

            except Exception as e:
                self.util.error(f"Feature extraction failed: {str(e)}")
                raise RuntimeError(f"Feature extraction failed: {str(e)}")

        else:
            self.util.debug(f"Reusing extracted OpenSMILE features from: {storage}")
            try:
                self.df = self.util.get_store(storage, store_format)
            except Exception as e:
                self.util.error(f"Failed to load stored features: {str(e)}")
                raise RuntimeError(f"Failed to load stored features: {str(e)}")

        return self.df

    def extract_sample(self, signal: np.ndarray, sr: int) -> np.ndarray:
        """Extract features from a single audio sample.

        Args:
            signal (np.ndarray): Audio signal as numpy array
            sr (int): Sample rate of the audio signal

        Returns:
            np.ndarray: Extracted features as numpy array

        Raises:
            ValueError: If signal or sample rate is invalid
        """
        if signal is None or len(signal) == 0:
            raise ValueError("Empty or invalid audio signal provided")

        if sr <= 0:
            raise ValueError(f"Invalid sample rate: {sr}")

        try:
            smile = opensmile.Smile(
                feature_set=self.feature_set,
                feature_level=opensmile.FeatureLevel.Functionals,
            )
            feats = smile.process_signal(signal, sr)
            return feats.to_numpy()
        except Exception as e:
            self.util.error(f"Failed to extract features from sample: {str(e)}")
            raise RuntimeError(f"Failed to extract features from sample: {str(e)}")

    def filter_features(self, feature_list: List[str] = None) -> pd.DataFrame:
        """Filter the extracted features to keep only the specified ones.

        Args:
            feature_list (List[str], optional): List of feature names to keep.
                If None, uses the list from config.

        Returns:
            pd.DataFrame: Filtered features DataFrame
        """
        # First ensure we're only using features indexed in the target dataframes
        self.df = self.df[self.df.index.isin(self.data_df.index)]

        if feature_list is None:
            try:
                # Try to get feature list from config
                import ast

                feature_list = ast.literal_eval(
                    glob_conf.config["FEATS"]["os.features"]
                )
            except (KeyError, ValueError, SyntaxError):
                self.util.debug("No feature list specified, using all features")
                return self.df

        if not feature_list:
            return self.df

        self.util.debug(f"Selecting features from OpenSMILE: {feature_list}")
        sel_feats_df = pd.DataFrame(index=self.df.index)
        hit = False

        for feat in feature_list:
            try:
                sel_feats_df[feat] = self.df[feat]
                hit = True
            except KeyError:
                self.util.warning(f"Feature '{feat}' not found in extracted features")

        if hit:
            self.df = sel_feats_df
            self.util.debug(f"New feature shape after selection: {self.df.shape}")

        return self.df

    @staticmethod
    def get_available_feature_sets() -> List[str]:
        """Get a list of available OpenSMILE feature sets.

        Returns:
            List[str]: List of available feature sets
        """
        return Opensmileset.AVAILABLE_FEATURE_SETS
