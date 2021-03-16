import abc
import pandas as pd

from pathlib import Path
from typing import Dict, Callable

from health_stage_classification.health_stage_classifiers import ahmad_et_al_2019, li_et_al_2019


class HealthStageClassifier(abc.ABC):
    def __init__(self, name):
        self.name = name

    @abc.abstractmethod
    def _determine_one_FPT(self, one_feature_df) -> int:
        pass

    def cut_FPTs_of_dataframe(self, dfs_to_cut: pd.DataFrame, labels_to_cut: pd.Series,
                              feature_dfs: pd.DataFrame) -> (pd.DataFrame, pd.Series):
        dfs_to_cut = dfs_to_cut.copy()
        labels_to_cut = labels_to_cut.copy()
        assert feature_dfs.index.nlevels > 1
        bearings = feature_dfs.index.levels[0]

        fpts: Dict[str, int] = {}
        for bearing in bearings:
            one_feature_df = feature_dfs.loc[bearing]
            fpts[bearing] = self._determine_one_FPT(one_feature_df=one_feature_df)
        for bearing in bearings:
            dfs_to_cut = dfs_to_cut.drop([(bearing, x) for x in range(fpts.get(bearing, 0))])
            labels_to_cut = labels_to_cut.drop([(bearing, x) for x in range(fpts.get(bearing, 0))])
        return dfs_to_cut, labels_to_cut

    def cut_FPTs_of_dataframe_dict(self, dfs_to_cut: Dict[str, pd.DataFrame], labels_to_cut: Dict[str, pd.Series],
                                   feature_dfs: pd.DataFrame) -> (
                                    Dict[str, pd.DataFrame], Dict[str, pd.Series], Dict[str, int]):
        dfs_to_cut = dfs_to_cut.copy()
        labels_to_cut = labels_to_cut.copy()
        bearings = dfs_to_cut.keys()

        fpts: Dict[str, int] = {}
        for bearing in bearings:
            one_feature_df = feature_dfs.loc[bearing]
            fpts[bearing] = self._determine_one_FPT(one_feature_df=one_feature_df)
        for bearing in bearings:
            dfs_to_cut[bearing] = dfs_to_cut[bearing].drop([x for x in range(fpts.get(bearing, 0))])
            labels_to_cut[bearing] = labels_to_cut[bearing].drop([x for x in range(fpts.get(bearing, 0))])
        return dfs_to_cut, labels_to_cut, fpts


class AhmadEtAl2019HealthStageClassifier(HealthStageClassifier):
    def __init__(self):
        HealthStageClassifier.__init__(self, "Ahmad Classifier")

    def _determine_one_FPT(self, one_feature_df) -> int:
        root_mean_square: pd.Series = one_feature_df["h_root_mean_square"]
        return ahmad_et_al_2019(root_mean_square=root_mean_square)


class LiEtAl2019HealthStageClassifier(HealthStageClassifier):
    def __init__(self):
        HealthStageClassifier.__init__(self, "Li et al. Classifier")

    def _determine_one_FPT(self, one_feature_df) -> int:
        kurtosis: pd.Series = one_feature_df["h_kurtosis"]
        return li_et_al_2019(kurtosis=kurtosis)
