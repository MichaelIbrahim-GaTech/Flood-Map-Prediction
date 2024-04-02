import xarray as xr
from deap import base, creator, tools, algorithms
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, balanced_accuracy_score, precision_recall_curve, average_precision_score, brier_score_loss, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import random
from sklearn.model_selection import cross_val_score
from typing import Union, Tuple, List, Optional, Any
import os
from matplotlib.colors import ListedColormap, BoundaryNorm
import src.Utils as utils
import matplotlib.colors as colors
import math
import matplotlib.patches as mpatches
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from catboost import Pool, EShapCalcType, EFeaturesSelectionAlgorithm
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline, make_pipeline

class BaseLineModel:
    """Base class for the baseline model.
    """
    number_of_instances = 0
    
    def __init__(self, 
                 labels_path:str, # path to the labels xarray dataset
                 dynamic_features_path:str = None, # path to the dynamic features xarray dataset
                 static_features_path:str = None, # path to the static features xarray dataset
                 replace_nan:bool=True, # replace nan values in labels by a value (useful when nan indicates no flood)
                 nan_value:int=0, # value to replace nan values in labels
                 reduce_train:bool=False, # reduce the train to take only band with flood 
                 train_start:str = "2002-08-03", # date where to split train test
                 train_end:str = "2003-01-1", # date where to split train test
                 test_start:str = "2003-01-01", # date where to split train test
                 test_end:str = "2003-03-17", # date where to split train test
                 val_start:str = "2003-01-1", # date where to split train test
                 val_end:str = "2003-03-17", # date where to split train test
                 inf_start:str = "2003-11-1", # date where to split train test
                 inf_end:str = "2004-01-01", # date where to split train test
                 name:str = "Model_01_default",
                 seed:int = 42,
                 is_cross_val:bool = True,
                 eval_metric_GA:str = "roc_auc"):
        """Constructor for the baseline model

        Args:
            labels_path (str): path to the labels xarray dataset
            dynamic_features_path (str, optional): path to the dynamic features xarray dataset. Defaults to None.
            static_features_path (str, optional): path to the static features xarray dataset. Defaults to None.
            replace_nan (bool, optional): replace nan values in labels by a value (useful when nan indicates no flood).
            nan_value (int, optional): value to replace nan values in labels. Defaults to 0.
            reduce_train (bool, optional): reduce the train to take only band with flood. Defaults to False.
            train_start (str, optional): date where to split train test. Defaults to "2002-08-03".
            train_end (str, optional): date where to split train test. Defaults to "2003-01-1".
            test_start (str, optional): date where to split train test. Defaults to "2003-01-01".
            test_end (str, optional): date where to split train test. Defaults to "2003-03-17".
            val_start (str, optional): date where to split train test. Defaults to "2003-01-1".
            val_end (str, optional): date where to split train test. Defaults to "2003-03-17".
            inf_start (str, optional): date where to split train test. Defaults to "2003-11-1".
            inf_end (str, optional): date where to split train test. Defaults to "2004-01-01".
            name (str, optional): . Defaults to "Model_01_default".
            seed (int, optional): Path to the dynamic features Xarray. Defaults to 42.
            is_cross_val (bool, optional): . Defaults to True.
            eval_metric_GA (str, optional): _description_. Defaults to "roc_auc".

        Raises:
            ValueError: _description_
        """

        self.eval_metric_GA = eval_metric_GA
        self.is_cross_val = is_cross_val
        self.seed = seed
        self.obj_name = name
        self.full_grid_all = None
        self.dataset_limits = {
            'train': {'start':  pd.to_datetime(train_start), 'end':  pd.to_datetime(train_end)},
            'val': {'start':  pd.to_datetime(val_start), 'end':  pd.to_datetime(val_end)},
            'test': {'start':  pd.to_datetime(test_start), 'end':  pd.to_datetime(test_end)},
            'all': {'start':  pd.to_datetime(train_start), 'end':  pd.to_datetime(test_end)},
            'inf': {'start':  pd.to_datetime(inf_start), 'end':  pd.to_datetime(inf_end)}
        }
        
        if (labels_path is None) | (
            (dynamic_features_path is None) & (static_features_path is None)
        ):
            raise ValueError("You must provide a path to the labels and either the dynamic or static features")

        self.labels = xr.open_dataset(labels_path)

        # replace nan values with "nan_value" in labels useful when nan indicates no flood
        #nan_mask = np.isnan(self.labels['__xarray_dataarray_variable__'].data)
        nan_mask = np.isnan(self.labels['__xarray_dataarray_variable__'].values)
        if replace_nan:
            self.labels['__xarray_dataarray_variable__'].data[nan_mask] = nan_value

        # load dynamic and static features and information about the number of features
        self.nb_feature_dynamic = 0
        self.all_dynamic_vars = []
        if dynamic_features_path is not None:
            self.dynamic_features = xr.open_dataset(dynamic_features_path)
            self.nb_feature_dynamic = len(self.dynamic_features.data_vars)
            self.all_dynamic_vars = list(self.dynamic_features.data_vars)
            self.nb_feature_dynamic = 2
            self.all_dynamic_vars = [self.all_dynamic_vars[0], self.all_dynamic_vars[1]]

        self.nb_feature_static = 0
        if static_features_path is not None:
            self.static_features = xr.open_dataset(static_features_path)
            self.all_static_vars = list(self.static_features.data_vars)
            self.nb_feature_static = len(self.static_features.data_vars)
            #print(self.static_features.data_vars)

        self.nb_feature = self.nb_feature_dynamic + self.nb_feature_static

        # dimensions of the dataset
        self.x_dim = self.labels.dims['x']
        self.y_dim = self.labels.dims['y']
        self.time_dim = self.labels.dims['time']

        # split train test mode
        self.reduce_train=reduce_train
        # train test split date
    

    def index_splitter(self, 
                       input_list:List[int], 
                       histodept_list_tp:list[int],
                       histodept_list_t2m:list[int],
                       use_derivative:bool)->Tuple[List[int], List[int]]:
        """ convert a list of indices into a list of static indices and a list of dynamic indices
        using the info about the structure of the dataset. Also add the derivative of the dynamic features if needed.
        And compute the name of the dynamic features.

        Args:
            input_list (List[int]): list of indices of the features to use (static and dynamic)
            histodept_list_tp (List[int]): history depth for averaging the dynamic features
            histodept_list_t2m (List[int]): history depth for averaging the dynamic features
            use_derivative (bool): Activate the computation of the derivative of the dynamic features

        Returns:
            Tuple[List[int], List[int]]: list of static indices and list of dynamic indices
        """

        self.all_dynamic_vars_names = []
        static = [x for x in input_list if 0 <= x < self.nb_feature_static]
        dynamic = [x for x in input_list if self.nb_feature_static <= x < self.nb_feature_static + self.nb_feature_dynamic]

        i_dyna = self.nb_feature_static

        dynamic = [x - self.nb_feature_static for x in dynamic]
        dynamic_names = [self.all_dynamic_vars[x] for x in dynamic]
        self.selected_features = static.copy()
        
        dyna_var = 'tp'
        for histodept in histodept_list_tp:
            self.all_dynamic_vars_names.append(dyna_var+"_"+str(histodept))
            self.selected_features.append(i_dyna)
            i_dyna = i_dyna +1
            if use_derivative:
                self.all_dynamic_vars_names.append(dyna_var+"_"+str(histodept)+"_deriv")
                self.selected_features.append(i_dyna)
                i_dyna = i_dyna +1
        dyna_var = 't2m'
        for histodept in histodept_list_t2m:
            self.all_dynamic_vars_names.append(dyna_var+"_"+str(histodept))
            self.selected_features.append(i_dyna)
            i_dyna = i_dyna +1
            if use_derivative:
                self.all_dynamic_vars_names.append(dyna_var+"_"+str(histodept)+"_deriv")
                self.selected_features.append(i_dyna)
                i_dyna = i_dyna +1
        return static, dynamic # list of static and dynamic indices used in the model (feature selection)
    

    

    def compute_total_features(self, 
                               dynamic_indices:List[int], 
                               histodept_list_tp:List[int],
                               histodept_list_t2m:List[int],
                               num_statics:int, 
                               use_derivative:bool)->int:
        """ Compute the total number of features used in the model, 
        including the constructed derivative of the dynamic features.

        Args:
            dynamic_indices (): dynamic features indices
            histodept (int): number of layers to average for the dynamic features
            histodept2 (int): number of layers to average for the dynamic features (second part)
            num_statics (int): number of static features
            use_derivative (int): compute the derivative of the dynamic features

        Returns:
            int : total number of features used in the model
        """
        total_features = num_statics
        total_features += len(histodept_list_tp)
        if use_derivative & (sum([h>1 for h in histodept_list_tp])>0):
            total_features += sum([h>1 for h in histodept_list_tp])
        total_features += len(histodept_list_t2m)
        if use_derivative & (sum([h>1 for h in histodept_list_t2m])>0):
            total_features += sum([h>1 for h in histodept_list_t2m])
            
        return total_features

    def create_feature_array(self, 
                             total_features:int)->Tuple[np.ndarray, int]:
        """ Create the array of features

        Args:
            total_features (int): total number of features used in the model

        Returns:
            Tuple[np.ndarray, int]: array of features.
        """
        return np.empty((self.x_dim * self.y_dim, total_features)), 0

    def populate_feature_array(self, 
                               feature_array: np.ndarray, 
                               feature_index: int,
                               static_indices: List[int], 
                               dynamic_indices: List[int], 
                               filtered_dynamic_features: xr.Dataset, 
                               histodept_list_tp: List[int],
                               histodept_list_t2m: List[int],
                               use_derivative:bool)->Tuple[np.ndarray, int]:
        """ Populate the array of features

        Args:
            feature_array (np.ndarray): array of features
            feature_index (int): index of the feature in the array
            static_indices (List[int]): list of static indices
            dynamic_indices (List[int]): list of dynamic indices
            filtered_dynamic_features (xr.Dataset): filtered dynamic features
            histodept_list (List[int]): history depth for averaging the dynamic features
            use_derivative (bool): compute the derivative of the dynamic features 

        Returns:
            Tuple[np.ndarray, int]: _description_
        """
        for static_index in static_indices:
            var = self.all_static_vars[static_index]
            feature_array[:, feature_index] = self.static_features[var].data.reshape(self.x_dim * self.y_dim)
            feature_index += 1

        var = 'tp'
        for histodept in histodept_list_tp:
            feature_index, feature_array = self.process_dynamic_feature(filtered_dynamic_features[var], histodept, feature_array, feature_index, use_derivative)
        var = 't2m'
        for histodept in histodept_list_t2m:
            feature_index, feature_array = self.process_dynamic_feature(filtered_dynamic_features[var], histodept, feature_array, feature_index, use_derivative)

        return feature_array, feature_index

    def create_target(self, 
                      time:str)->np.ndarray:
        """ Reshape the target array
        Args:
            time (str): date of the target
        Returns:
            np.ndarray: target array
        """
        return self.labels['__xarray_dataarray_variable__'].sel(time=time).data.reshape(-1)

    def get_features_and_targets(self, indices, use_derivative, histodept_list_tp, histodept_list_t2m):
        features_all_bands = []
        targets_all_bands = []
        static_indices, dynamic_indices = self.index_splitter(indices, histodept_list_tp, histodept_list_t2m, use_derivative)
        num_statics = len(static_indices)

        for band in self.labels.time.values:
            start_date = pd.to_datetime(band)
            filtered_dynamic_features = self.dynamic_features.sel(time=slice(None, start_date))

            total_features = self.compute_total_features(dynamic_indices, histodept_list_tp, histodept_list_t2m, num_statics, use_derivative)
            feature_array, feature_index = self.create_feature_array(total_features)
            feature_array, feature_index = self.populate_feature_array(feature_array, feature_index, static_indices, dynamic_indices, filtered_dynamic_features, histodept_list_tp, histodept_list_t2m, use_derivative)

            features_all_bands.append(feature_array if total_features > 0 else None)
            targets_all_bands.append(self.create_target(band))

        return features_all_bands, targets_all_bands


    def train_flood_prediction_model(self, 
                                     indices:List[int], 
                                     histodept_list_tp: List[int],
                                     histodept_list_t2m: List[int],
                                     use_derivative:bool)->Tuple[List[CatBoostClassifier], float]:
        """Train the random forest model

        Args:
            indices (List[int]): list of indices of the features to use (static and dynamic)
            histodept_list (List[int]): history depth for averaging the dynamic features
            use_derivative (bool): compute the derivative of the dynamic features

        Returns:
            Tuple[CatBoostClassifier, float]: Trained random forest model and the score of the model
        """

        features_all_bands, targets_all_bands = self.get_features_and_targets(indices, use_derivative, histodept_list_tp, histodept_list_t2m)

        # split train test
        temp_data = {'train': {'X':  None, 'y':  None}, 
                     'val': {'X':  None, 'y':  None}, 
                     'test': {'X':  None, 'y':  None}, 
                     'all': {'X':  None, 'y':  None}, 
                     'inf': {'X':  None, 'y':  None}}
        
        for dataset in ['train', 'val', 'test', 'all', 'inf']:
            if dataset in ['all','inf']:
                france_only = False
            else:
                france_only = True
            temp_data[dataset]['X'], temp_data[dataset]['y'] = self.split_subset_date(
                features_all_bands,
                targets_all_bands,
                start_date = self.dataset_limits[dataset]['start'],
                end_date = self.dataset_limits[dataset]['end'],
                france_only = france_only)
            
        #folds = 5
        #skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        ##for fold, (train_idx, val_idx) in enumerate(skf.split(temp_data['train']['X'], temp_data['train']['y'])):
        #results = {}
        #eliminated = set()
        #for features_count in range(10,35):
        #    n = temp_data['train']['X'].shape[0]
        #    val_idx = [i for i in range(n) if i*folds//n == 0]
        #    train_idx = [i for i in range(n) if i*folds//n != 0]
        #    train_pool = Pool(temp_data['train']['X'][train_idx,:], temp_data['train']['y'][train_idx])
        #    test_pool = Pool(temp_data['train']['X'][val_idx,:], temp_data['train']['y'][val_idx])
        #    model = CatBoostClassifier(iterations=100, verbose=1000,random_seed=42)
        #    summary = model.select_features(
        #        train_pool,
        #        eval_set=test_pool,
        #        features_for_select=list(range(temp_data['train']['X'].shape[1])),     # we will select from all features
        #        num_features_to_select=features_count,  # we want to select exactly important features
        #        steps=10,                                     # more steps - more accurate selection#
        #        shap_calc_type=EShapCalcType.Regular,            # can be Approximate, Regular and Exact
        #        train_final_model=True,                          # to train model with selected features
        #        logging_level='Silent',
        #        plot=False
        #    )
        #    print('Selected important features: {}'.format(
        #        len(summary['selected_features'])
        #    ))
        #    
        #    features = []
        #    for i in summary['selected_features']:
        #        features.append(self.selected_features[i])
        #    
        #    models = []
        #    folds = 5
        #    y_solution = np.zeros(shape=(temp_data['train']['X'].shape[0],1))
        #    for fold in range(folds):
        #        n = temp_data['train']['X'].shape[0]
        #        val_idx = [i for i in range(n) if i*folds//n == fold]
        #        train_idx = [i for i in range(n) if i*folds//n != fold]
        #        model = CatBoostClassifier(iterations=100, verbose=1000,random_seed=42)#, eval_metric='AUC')
        #        model.fit(temp_data['train']['X'][train_idx,:][:,features], temp_data['train']['y'][train_idx], eval_set=(temp_data['train']['X'][val_idx,:][:,features], temp_data['train']['y'][val_idx]), use_best_model=True)
        #        preds = model.predict_proba(temp_data['train']['X'][val_idx,:])[:, 1]
        #        y_solution[val_idx,0] = preds
        #        models.append(model)
        #    
        #    y_pred_proba_val = utils.batch_predict(models,temp_data['test']['X'], is_proba = True)
        #    y_pred_val = utils.batch_predict(models,temp_data['test']['X'], is_proba = False)
        #    score = self.compute_metric(y_pred_proba_val,
        #                                    y_pred_val,
        #                                    temp_data['test']['y'], 
        #                                    mode='roc')
        #    print(f"validatin roc : {score}")
        #    results[features_count] = score
        #    
        #    print(results)
        ##print(summary)
        #print(self.selected_features)
        #for i in summary['eliminated_features']:
        #    print(i, self.getfname(self.selected_features[i]))
    
        models = []
        folds = 5
        y_solution = np.zeros(shape=(temp_data['train']['X'].shape[0],1))
        skf = StratifiedKFold(n_splits=folds, random_state=42, shuffle=True)
        #for fold, (train_idx, val_idx) in enumerate(skf.split(temp_data['train']['X'], temp_data['train']['y'])):
        for fold in range(folds):
            n = temp_data['train']['X'].shape[0]
            val_idx = [i for i in range(n) if i*folds//n == fold]
            train_idx = [i for i in range(n) if i*folds//n != fold]
            
            model2 = CatBoostClassifier(iterations=100, verbose=1000,random_seed=42,learning_rate=0.05,use_best_model=True)#, eval_metric='BrierScore')
            model2.fit(temp_data['train']['X'][train_idx,:], temp_data['train']['y'][train_idx], eval_set=(temp_data['train']['X'][val_idx,:], temp_data['train']['y'][val_idx]), use_best_model=True)
            
            preds = model2.predict_proba(temp_data['train']['X'][val_idx,:])[:, 1]
            y_solution[val_idx,0] = preds
            models.append(model2)
        
                
        scores = roc_auc_score(temp_data['train']['y'], y_solution)

        
        # keep the model and the data if needed (sometime we just want to know the AUC without saving the model)
        self.save_temp_data_to_PV(temp_data,models)

        return (models, 
                scores.mean())


    def save_temp_data_to_PV(self,temp_data,model):
        """ Save the model and the data if needed (sometime we just want to know the AUC without saving the model)

        Args:
            temp_data (_type_): flatten data
            model (_type_): trained model
        """
        self.model = model
        self.X_train = temp_data['train']['X']
        self.X_val = temp_data['val']['X']
        self.X_test = temp_data['test']['X']
        self.X_all = temp_data['all']['X']
        self.X_inf = temp_data['inf']['X']
        self.y_train =  temp_data['train']['y']
        self.y_val = temp_data['val']['y']
        self.y_test = temp_data['test']['y']
        self.y_all = temp_data['all']['y']

    def process_dynamic_feature(self, 
                                feature: xr.DataArray, # dynamic feature
                                histodepth: int, # history depth for averaging the dynamic features
                                feature_array: np.ndarray, # array of features
                                feature_index: int, # index of the feature in the array
                                use_derivative: bool # compute the derivative of the dynamic features
                                )->Tuple[int, np.ndarray]:
        """
        Process a dynamic feature by averaging histodepth layers also compute the derivative if needed.
        Args:
            feature (xr.DataArray): dynamic feature
            histodepth (int): history depth for averaging the dynamic features
            feature_array (np.ndarray): array of features
            feature_index (int): index of the feature in the array
            use_derivative (bool): compute the derivative of the dynamic features

        Returns:
            Tuple[int, np.ndarray]: index of the feature in the array and the array of features
        """
        if histodepth > 0:
            last_layers = feature.isel(time=slice(-histodepth, None))
            averaged_feature = last_layers.mean(dim='time')
            feature_array[:, feature_index] = averaged_feature.data.reshape(self.x_dim * self.y_dim)
            feature_index += 1

            if use_derivative and histodepth > 1:
                average_derivative = np.gradient(last_layers, axis=0).mean(axis=0)
                feature_array[:, feature_index] = average_derivative.reshape(self.x_dim * self.y_dim)
                feature_index += 1

        return feature_index, feature_array

    def remove_non_france_data(self, 
                               X: np.ndarray, 
                               y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Filters out data not relevant for France.
        (data inside France but outside France's borders are labeled as -1)

        Parameters
        ----------
        X : np.ndarray
            The feature array.
        y : np.ndarray
            The target array.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The filtered feature and target arrays.
        """
        valid_indices = (y != -1)
        return X[valid_indices], y[valid_indices]


    def print_feature_importance(self):
        """Prints the importance of each selected feature in the model.
        """
        importances = np.zeros_like(np.array(self.model[0].feature_importances_))
        for m in self.model:
            importances = importances + np.array(m.feature_importances_)
        importances = list(importances)
        feature_importances = zip(self.selected_features, importances)
        sorted_feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

        feature_names = []
        for feature, importance in sorted_feature_importances:
            feature_names.append(self.getfname(feature))
            print(f"{self.getfname(feature)}: {importance}")

        features, importance_values = zip(*sorted_feature_importances)
        plt.figure(figsize=(12, 6))
        bars = plt.bar(feature_names, importance_values, capsize=5, color='skyblue', edgecolor='black', alpha=0.7)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title('Feature Importance')
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability


        plt.tight_layout()
        plt.show()

    def getfname(self, 
                 index: int) -> str:
        """Retrieves the feature name for the given index.

        Parameters
        ----------
        index : int
            The index of the feature.

        Returns
        -------
        str
            The name of the feature corresponding to the provided index.
        """    
        
        var_names = self.all_static_vars + self.all_dynamic_vars_names

#        reduce_list = [var_names[i] for i in self.selected_features]
        
        return var_names[index]

    def get_flood_bands(self, 
                        all_bands: List[str], 
                        labels: xr.Dataset) -> List[str]:
        """ Get the bands with flood, use to reduce the train dataset to only the bands with flood 
        when (reduce_train=True) by default we keep all the bands.

        Args:
            all_bands (List[str]): list of all the bands
            labels (xr.Dataset): labels dataset
        Returns:
            List[str]: list of bands with flood
        """
        flood_bands = []
        for band_date in all_bands:
            band_data = labels['__xarray_dataarray_variable__'].sel(time=band_date).values
            if np.any(band_data != 0):
                flood_bands.append(band_date)
        return flood_bands

    def split_subset_date(self, 
                          features_filtered: List[np.ndarray], 
                          target_source: List[np.ndarray], 
                          start_date:str="2003-09-01", 
                          end_date:str="2003-09-01",
                          france_only:bool = False,
                          is_inf:bool = False)-> Tuple[np.ndarray, np.ndarray]:
        """_summary_

        Args:
            features_filtered (List[np.ndarray]): Filtered features array.
            target_source (List[np.ndarray]): Filtered target array.
            start_date (str, optional): start date for the subset. Defaults to "2003-09-01".
            end_date (str, optional): end date for the subset. Defaults to "2003-09-01".
            france_only (bool, optional): Filtering out data not relevant for France. Defaults to False.
            is_inf (bool, optional): is it the inference dataset (if so there is no labels). Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray]: filtered feature and label arrays
        """
        start, end = pd.to_datetime(start_date), pd.to_datetime(end_date)
        all_bands = self.labels.time.values

        if self.reduce_train:
            all_bands = self.get_flood_bands(all_bands, self.labels)

        bands = [band for band in all_bands if start < band <= end]
        band_to_index = {band: idx for idx, band in enumerate(self.labels.time.values)}

        num_samples = len(bands)
        feature_shape = features_filtered[0].shape[:2]

        X_tmp = np.empty((num_samples, *feature_shape))
        if ~is_inf:
            y_tmp = np.empty((num_samples, features_filtered[0].shape[0]))

        for band in bands:
            index = band_to_index[band]
            feature = features_filtered[index]
            if ~is_inf:
                target = target_source[index]
            idx = bands.index(band)
            X_tmp[idx] = feature.reshape(*feature_shape)
            if ~is_inf:
                y_tmp[idx] = target

        if is_inf:
            y = None
        else:
            y = y_tmp.reshape(-1)
        X = np.reshape(X_tmp, (X_tmp.shape[0] * X_tmp.shape[1], X_tmp.shape[2]))
        
        if france_only:
            X, y = self.remove_non_france_data(X, y)

        return X, y


    def remove_nan(self,
                   X: np.ndarray,
                   y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Filters out data with nan values.

        Args:
            X (np.ndarray): feature array
            y (np.ndarray): label array

        Returns:
            Tuple[np.ndarray, np.ndarray]: filtered feature and label arrays
        """
        valid_indices = ~np.isnan(y) 
        return X[valid_indices], y[valid_indices]

    def train_model(self,
                   individual: List[Any])-> float:
        """Transform GA individual into model parameters

        Args:
            individual (List[]): GA individual
        
        Returns:
            float: score of the model
        """
        print("train_model",individual)
        return self.check_load(indices = [i for i, use in enumerate(individual[:self.nb_feature]) if use], 
                                  histodept_list_tp = individual[self.nb_feature+1],
                                  histodept_list_t2m = individual[self.nb_feature+2],
                                  use_derivative = individual[self.nb_feature]) 


    def check_load(self, 
                   indices: List[int], 
                   histodept_list_tp: List[int],
                   histodept_list_t2m: List[int],
                   use_derivative: bool, 
                   debug:bool=False):
        """ Check if the individual is valid and launch the training of the model if it is valid.
        Set the auc to 0 if the individual is not valid (useful for the GA)

        Args:
            indices (_type_): selected features indices
            use_derivative (_type_): _description_
            debug (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        if debug:
            print("indices",indices)
            print("use_derivative",use_derivative)
        
        if(len(indices)>0):
            model, score = self.train_flood_prediction_model(
                indices, 
                histodept_list_tp,
                histodept_list_t2m,
                use_derivative)
        else:
            model = None
        if model is None:
            score = 0
        print("score",score)

        return model,score


    def predict_prob(self, dataset = "Train"):
        """ Predict the probability of flood for the train, val, test, all or inference dataset

        Args:
            dataset (str, optional): . Defaults to "Train".

        Returns:
            _type_: _description_
        """
        if dataset == "Train":
            X = self.X_train
        elif dataset == "Test":
            X = self.X_test
        elif dataset == "Val":
            X = self.X_val
        elif dataset == "All":
            X = self.X_all
        elif dataset == "Inf":
            X = self.X_inf
        else:
            raise ValueError("dataset must be either Val, Train, All or Test")
        
        y_pred_proba_test = np.zeros(shape=(1,X.shape[0]))
        for m in self. model:
            y_pred_proba_test = y_pred_proba_test + m.predict_proba(X)[:, 1]/4
            
        return y_pred_proba_test.T

    def compute_full_grid(self):
        """Computes a grid of predicted probabilities alongside the corresponding labels on the full resolution dataset.

        Returns
        -------
        np.ndarray
            The full grid of predicted probabilities.
        """
        factor = self.y_dim * self.x_dim

        all_size = int(self.y_all.shape[0]/factor)
        self.full_grid_all = np.full((all_size, self.y_dim, self.x_dim), np.nan)
        full_grid_flat = self.full_grid_all.reshape(-1)
        full_grid_flat = self.predict_prob("All")
        self.full_grid_all = full_grid_flat.reshape(all_size, self.y_dim, self.x_dim)

        inf_size = int(self.X_inf.shape[0]/factor)
        self.full_grid_inf = np.full((inf_size, self.y_dim, self.x_dim), np.nan)
        full_grid_flat = self.full_grid_inf.reshape(-1)
        full_grid_flat = self.predict_prob("Inf")
        self.full_grid_inf = full_grid_flat.reshape(inf_size, self.y_dim, self.x_dim)

    def save_FP_FN_map(self, 
                            save_path = "graph/model1_AP/compare_threshold/",
                            thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.6]):
        """ Predictions map at different thresholds with false positive and false negative.

        Args:
            save_path (str, optional): Saving path. Defaults to "graph/model1_AP/compare_threshold/".
            thresholds (list, optional): Prediction threshold to define positive / negative predictions.
        """
        if self.full_grid_all is None:
            self.compute_full_grid()
        font_size = 32
        
        n_cols = 2
        n_rows = math.ceil((len(thresholds) + 2) / n_cols)
        cmap1 = plt.cm.viridis_r
        cmap1.set_bad('#A5E0E4', 1.)
        for k, band_index in enumerate(self.labels.sel(time=slice(
                            self.dataset_limits["train"]["start"],
                            self.dataset_limits["train"]["end"],
                            )).time.values):
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 25))
            label_map = self.labels['__xarray_dataarray_variable__'][k].values.copy()
            pred_map = self.full_grid_all[k, :, :].copy()
            
            water_mask = label_map == -1
            pred_map[water_mask] = np.nan
            label_map[water_mask] = np.nan

            
            axs[0,0].imshow(pred_map, cmap=cmap1, interpolation='none')
            axs[0,0].set_title('Predicted Flood Probabilities', fontsize=font_size)
            axs[0,0].axis('off')
            
            
            legend_elements = [mpatches.Patch(color=plt.cm.viridis(0), label='Flood'),
                            mpatches.Patch(color=plt.cm.viridis_r(0), label='No Flood')]

            axs[0,1].imshow(label_map, cmap=cmap1, interpolation='none')
            axs[0,1].set_title('Label', fontsize=font_size)
            axs[0,1].axis('off')
            axs[0,1].legend(handles=legend_elements, loc="upper right")

            legend_elements = [mpatches.Patch(color='white', label='TN'),
                            mpatches.Patch(color='grey', label='FN'),
                            mpatches.Patch(color='black', label='TP'),
                            mpatches.Patch(color='red', label='FP'),
                            mpatches.Patch(color='#A5E0E4', label='Water')]

            for i, threshtest in enumerate(thresholds):

                cmap2 = ListedColormap(['white', 'grey', 'black', 'red'])
                cmap2.set_bad('#A5E0E4', 1.)
                boundaries = [-0.5, 0.5, 1.5, 2.5, 3.5]
                norm = BoundaryNorm(boundaries, cmap2.N, clip=True)
                pred_map_at_th = pred_map.copy()

                pred_map_at_th[pred_map < threshtest] = 0
                pred_map_at_th[pred_map >= threshtest] = 1

                classification_results = np.zeros_like(pred_map_at_th)
                classification_results[(pred_map_at_th == 0) & (label_map == 0)] = 0  # TN
                classification_results[(pred_map_at_th == 1) & (label_map == 0)] = 1
                classification_results[(pred_map_at_th == 1) & (label_map == 1)] = 2
                classification_results[(pred_map_at_th == 0) & (label_map == 1)] = 3  # TP
                classification_results[water_mask] = np.nan

                xi = i // n_cols+1
                yi = i % n_cols
                axs[xi,yi].imshow(classification_results, cmap=cmap2, norm=norm, interpolation='none')
                axs[xi,yi].set_title(f"Threshold: {threshtest}", fontsize=font_size)
                axs[xi,yi].axis('off')
                axs[xi,yi].legend(handles=legend_elements, loc="upper right")


            plt.tight_layout()
            #fig.suptitle(f"Week : {utils.split_time_index(band_index)}", fontsize=font_size+10)
            isExist = os.path.exists(save_path)
            if not isExist:
                os.makedirs(save_path)
            plt.savefig(f"{save_path}{utils.split_time_index(band_index)}.png")

            plt.close(fig)


    def save_prediction_map_and_labels(self, 
                            save_path:str = "graph/model1_AP/label_and_pred/"):
        """ Predictions map at different thresholds with labels.

        Args:
            save_path (str, optional): Saving path. Defaults to "graph/model1_AP/label_and_pred/".
        """
        if self.full_grid_all is None:
            self.compute_full_grid()

        font_size = 32

        for k, band_index in enumerate(self.labels.sel(time=slice(
                            self.dataset_limits["train"]["start"],
                            self.dataset_limits["train"]["end"],
                            )).time.values):
            labelmap = self.labels['__xarray_dataarray_variable__'][k].values.copy()

            predictionmap = self.full_grid_all[k, :, :].copy()
            predictionmap[labelmap == -1] = np.nan

            labelmap[labelmap == -1] = np.nan

            fig, axs = plt.subplots(1, 2, figsize=(35, 18))
            cmap = plt.cm.gray_r
            cmap.set_bad('#A5E0E4', 1.)

            axs[0].imshow(predictionmap, cmap=cmap, interpolation='none')
            axs[0].set_title('M1 Flood Probabilities', fontsize=font_size)
            axs[0].axis('off')

            axs[1].imshow(labelmap, cmap=cmap, interpolation='none')
            axs[1].set_title(f'Label', fontsize=font_size)
            axs[1].axis('off')

            # Increase label size for axis ticks
            for ax in axs:
                ax.tick_params(axis='both', which='major', labelsize=font_size)

            plt.tight_layout()
            
            isExist = os.path.exists(save_path)
            if not isExist:
                os.makedirs(save_path)
            plt.savefig(f"{save_path}{utils.split_time_index(band_index)}.png")
            plt.close(fig)



    def save_prediction_map(self, 
                            save_path:str = "graph/model1_AP/predictions/"):
        """ Predictions map at different thresholds.

        Args:
            save_path (str, optional): Saving path. Defaults to "graph/model1_AP/predictions/".
        """
        font_size = 12

        if self.full_grid_all is None:
            self.compute_full_grid()
        cmap = plt.cm.viridis_r
        cmap.set_bad('#A5E0E4', 1.)
#       for specific_time_slice in range(self.full_grid_all.shape[0]):
        for k, band_index in enumerate(self.labels.sel(time=slice(
                            self.dataset_limits["train"]["start"],
                            self.dataset_limits["train"]["end"],
                            )).time.values):
            labelmap = self.labels['__xarray_dataarray_variable__'][k].values

            predictionmap = self.full_grid_all[k, :, :].copy()
            predictionmap[labelmap == -1] = np.nan

            plt.figure(figsize=(15, 10))
            plt.imshow(predictionmap, cmap=cmap, interpolation='none')
            plt.colorbar(label='M1 Flood Probability')
            plt.title(f'Flood Probabilities - Time Slice {band_index}',fontsize=font_size)
            plt.axis("off")

            isExist = os.path.exists(save_path)
            if not isExist:
                os.makedirs(save_path)
            plt.savefig(f"{save_path}{utils.split_time_index(band_index)}.png")
            plt.close()

    def save_error_map(self, 
                            save_path:str = "graph/model1_AP/save_error_map/"):
        """ Predictions map at different thresholds with labels.

        Args:
            save_path (str, optional): Saving path. Defaults to "graph/model1_AP/label_and_pred/".
        """
        font_size = 12

        if self.full_grid_all is None:
            self.compute_full_grid()

        for k, band_index in enumerate(self.labels.sel(time=slice(
                            self.dataset_limits["train"]["start"],
                            self.dataset_limits["train"]["end"],
                            )).time.values):
            labelmap = self.labels['__xarray_dataarray_variable__'][k].values.copy()

            predictionmap = self.full_grid_all[k, :, :].copy()
            predictionmap[labelmap == -1] = np.nan

            labelmap[labelmap == -1] = np.nan

            cmap_em = plt.cm.seismic
            cmap_em.set_bad('#A5E0E4', 1.)
            errormap = predictionmap - labelmap
            plt.figure(figsize=(15, 10))
            norm = colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

            plt.imshow(errormap, cmap=cmap_em, interpolation='none', norm=norm)

            plt.colorbar(label='M1 Flood Errors -1 : flood missed, 1 : false alarm')
            plt.title(f'Flood Probabilities - Time Slice {band_index}',fontsize=font_size)
            plt.axis("off")

            isExist = os.path.exists(save_path)
            if not isExist:
                os.makedirs(save_path)
            plt.savefig(f"{save_path}{utils.split_time_index(band_index)}.png")
            plt.close()



    def compute_all_metrics(self):
        """Compute all the metrics for the train, val and test dataset"""
        for dataset in ["Train", "Test", "Val"]:
            print(f"{dataset} :")
            if dataset == "Train":
                X = self.X_train
                y = self.y_train
            elif dataset == "Test":
                X = self.X_test
                y = self.y_test
            elif dataset == "Val":
                X = self.X_val
                y = self.y_val
            else:
                raise ValueError("dataset must be either Val, Train or Test")
            y_pred_proba_val = utils.batch_predict(self.model,X, is_proba = True)
            y_pred_val = utils.batch_predict(self.model,X, is_proba = False)

            for metric in ["roc", "BrierScore"]:
                score = self.compute_metric(y_pred_proba_val,
                                            y_pred_val,
                                            y, 
                                            mode=metric)
                print(f"{metric} : {score}")
            print(f"") 


    def compute_metric(self, 
                       y_pred_proba_val: np.ndarray,
                       y_pred_val: np.ndarray, 
                       y: np.ndarray, 
                       mode = "roc")->float:
        """ Compute a metric for the model

        Args:
            y_pred_proba_val (np.ndarray): model prediction probabilities
            y_pred_val (np.ndarray): model prediction
            y (np.ndarray): true labels
            mode (str, optional): metric to compute. Defaults to "roc".

        Returns:
            float: score of the metric
        """

        if mode == "roc":
            score = roc_auc_score(y, y_pred_proba_val)
        elif mode == "AP":
            score = average_precision_score(y, y_pred_proba_val)
        elif mode == "BrierScore":
            score = brier_score_loss(y, y_pred_proba_val)
        elif mode == "acc":
            score = accuracy_score(y, y_pred_val)
        elif mode == "bacc":
            score = balanced_accuracy_score(y, y_pred_val)
        elif mode == "f1":
            score = f1_score(y, y_pred_val)
        elif mode == "precision":
            score = precision_score(y, y_pred_val)
        elif mode == "recall":
            score = recall_score(y, y_pred_val)
        else:
            raise ValueError("mode must be either roc, acc, f1, precision or recall")
        return score  
    

    def process_full_predictions(self, 
                                 dataset="Full_Test"):
        """Computes the prediction from the succession of both Model 1 and Model 2.
        """
        if dataset == "Test":
            X = self.X_test[:,self.selected_features]
            y = self.y_test
            model1_score = self.X_test[:,-1] #M1_score column index
            model1_score = self.Full_X_test[:,-1] #M1_score column index
        elif dataset == "Val":
            X = self.X_val[:,self.selected_features]
            y = self.y_val
            model1_score = self.X_val[:,-1] #M1_score column index
        elif dataset == "Train":
            X = self.X_train[:,self.selected_features]
            y = self.y_train
            model1_score = self.X_train[:,-1] #M1_score column index
        else:
            raise ValueError("dataset must be either Val, Train or Test")
        rejected_predictions = model1_score < 0.5
        y_pred_proba_val = utils.batch_predict(self.model,X, is_proba = True)
        y_pred_proba_val[rejected_predictions] = 0
        y_pred_val = utils.batch_predict(self.model,X, is_proba = False)
        y_pred_val[rejected_predictions] = 0
        for eval_metric in ["roc", "AP", "BrierScore", "f1", "precision", "recall", "acc"]:
            score = self.compute_metric(y_pred_proba_val,
                                         y_pred_val,
                                         y, 
                                         rejected_predictions=rejected_predictions, 
                                         mode=eval_metric)
            print(f"{eval_metric} : {score}")

    def auc_graph(self, 
                  dataset:List[str] = ["Train"], 
                  metrics:List[str] = ["auto"],
                  key_thresholds:List[float] = [0.001,0.005,0.01,0.05,0.1, 0.2,0.3, 0.5, 0.9]):
        """ Plot the ROC curve and the AUC for the train or test dataset

        Args:
            dataset (List[str], optional): dataset to plot. Defaults to ["Train"].
            metrics (List[str], optional): metrics to plot. Defaults to "auto".
            key_thresholds (List[float], optional): Key thresholds to plot for the AUC-ROC

        """
        # plot the ROC curve and the AUC for the train or test dataset
        if len(dataset)==0:
            datasets = ["Train", "Test", "Val"]
        else:
            datasets = dataset
        if len(metrics)==0:
            metrics = [self.eval_metric_GA]
        else:
            metrics = [metrics]

        for dataset in datasets:
            for metric in metrics:
                if dataset == "Train":
                    print("Train")
                    X = self.X_train
                    y = self.y_train
                elif dataset == "Test":
                    print("Test")
                    X = self.X_test
                    y = self.y_test
                elif dataset == "Val":
                    print("Val")
                    X = self.X_val
                    y = self.y_val
                else:
                    raise ValueError("dataset must be either Val, Train or Test")
                
                y_pred_proba_test = y_solution = np.zeros(shape=(1,X.shape[0]))
                for m in self. model:
                    y_pred_proba_test = y_pred_proba_test + m.predict_proba(X)[:, 1]/4
                
                y_pred_proba_test = y_pred_proba_test.T
                print(y_pred_proba_test.shape)


                plt.figure(figsize=(8, 6))
                if metric == "roc_auc":
                    fpr, tpr, thresholds = roc_curve(y, y_pred_proba_test)
                    auc = roc_auc_score(y, y_pred_proba_test)
                    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.3f)' % auc)
                    plt.xlabel(f'False Positive Rate {dataset}')
                    plt.ylabel(f'True Positive Rate {dataset}')
                    plt.title(f'Receiver Operating Characteristic {dataset}')

                elif  metric == "average_precision":
                    precision, recall, thresholds = precision_recall_curve(y, y_pred_proba_test)
                    auc =  average_precision_score(y, y_pred_proba_test)
                    plt.plot([0, 1], [1, 0], color='navy', lw=2, linestyle='--')
                    plt.plot(precision, recall, color='darkorange', lw=2, label='PR curve (area = %0.3f)' % auc)
                    plt.xlabel(f'Recall {dataset}')
                    plt.ylabel(f'Precision {dataset}')
                    plt.title(f'Precision Recall Curve {dataset}')


                if metric == "roc_auc":
                    for thresh in key_thresholds:
                        idx = np.where(thresholds >= thresh)[0][-1]
                        plt.plot(fpr[idx], tpr[idx], 'o', markersize=10, label=f'Threshold {thresh:.3f}')
            

                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.0])
                plt.legend(loc="lower right")
                plt.show()

                print(f"AUC: {auc}")
    
    def save_to_disk(self, 
                     name:str = ""):
        """Save the model to disk

        Args:
            name (bool, optional): Name of the model.
        """
        if name == "":
            name = self.obj_name

        with open(self.obj_name+'.pkl', 'wb') as file: 
            pickle.dump(self, file)



    def load_from_disk(self, 
                       name:str)->object:
        """ Load the model from disk

        Args:
            name (str): Name of the model.

        Returns:
            object: model
        """
        with open(name+'.pkl', 'rb') as file:  
            loaded = pickle.load(file)

        self = loaded

        return loaded

    
