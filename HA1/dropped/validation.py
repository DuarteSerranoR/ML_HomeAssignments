import numpy as np
from enum import Enum
from sklearn import tree
from itertools import product


class ModelTypes(Enum):

    regression = 0
    classification_binary = 1
    classification_n_ary = 2

class CrossValidationMethods(Enum):

    simple = 0
    n_fold = 1
    leave_one_out = 2

# TODO - retrospective validation
# TODO - KFold cross validation

def simple_cross_validation(*, model, x, y, train_partition=(0, 0.8), model_type: ModelTypes):
    n, m = x.shape
    
    start = int(train_partition[0] * n)
    end = int(train_partition[1] * n)
    
    x_train = x[start:end, :]
    y_train = y[start:end]
    x_test = np.concatenate((x[:start, :], x[end:, :]))
    y_test = np.concatenate((y[:start], y[end:]))
    
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    if model_type == ModelTypes.regression
    return {metric_name: metric_f(y_test, y_pred) for (metric_name, metric_f) in metrics.items()}


def kfold(*, model, x, y, k=5, model_type: ModelTypes):
    n, m = x.shape
    results = []
    
    for i in range(k):
        start = i * (1/k)
        end = start + (1/k)
        if (i + 1) == k:
            end = 1.0
            
        result = simple_cross_validation(model=model, x=x, y=y, train_partition=(start, end), model_type: ModelTypes)
        
        results.append(result)
    
    return results


# TODO - Simple cross validation
# TODO - N-Fold Cross validation
# TODO - Leave-one-out cross validation

def model_selection(
        model_type: ModelTypes, # regression, classification_binary, classification_n_ary
        cross_validation_methodology: CrossValidationMethods, # simple, n_fold, leave_one_out
        target_classifier: str, # specifies which classifier should be focused on during validation
        x: np.ndarray, y: np.ndarray,
        dev_ratio = 0.1,
        kfold_k = 5,
        hyper_params_regression_criterions = [ "squared_error" ],
        hyper_params_regression_splitters = [ "best" ],
        hyper_params_regression_max_depths = None,
        hyper_params_regression_min_samples_splits = [ 2 ],
        hyper_params_regression_min_samples_leafs = [ 1 ],
        hyper_params_regression_min_weight_fraction_leaf = [ 0.0 ],
        hyper_params_regression_max_features = [ None ],
        hyper_params_regression_random_state = [ None ], # int
        hyper_params_regression_max_leaf_nodes = [ None ], # int
        hyper_params_regression_min_impurity_decrease = [ 0.0 ],
        hyper_params_regression_ccp_alpha = [ 0.0 ]
    ):
    """
    hyper_params = {
            "criterions": ["squared_error", "friedman_mse", "absolute_error","poisson"],
            "splitters": ["best", "random"],
            "max_depths": range(10,100,10),
            "max_features": ["auto", "sqrt", "log2"],
            "min_samples_leafs": range(1,30,1),
            "min_samples_splits": range(2,30,1)
        }):
    """

    if cross_validation_methodology != CrossValidationMethods.n_fold:
        train_size = x_init.shape[0] * train_ratio
        ivs_and_dev_size = (1 - train_ratio) / 2

        x_train = x_init[:train_size]
        x_dev = x_init[train_size:train_size+ivs_and_dev_size]
        x_ivs = x_init[train_size+ivs_and_dev_size:]

        y_train = y_init[:train_size]
        y_dev = y_init[train_size:train_size+ivs_and_dev_size]
        y_ivs = y_init[train_size+ivs_and_dev_size:]

    
    # Hyperparameters
    if model_type == ModelTypes.regression:
        criterion = hyper_params_regression_criterions
        splitter = hyper_params_regression_splitters
        max_depth = hyper_params_regression_max_depths
        min_samples_splits = hyper_params_regression_min_samples_splits
        min_samples_leaf = hyper_params_regression_min_samples_leafs
        min_weight_fraction_leaf = hyper_params_regression_min_weight_fraction_leaf
        max_features = hyper_params_regression_max_features
        random_state = hyper_params_regression_random_state
        max_leaf_nodes = hyper_params_regression_max_leaf_nodes
        min_impurity_decrease = hyper_params_regression_min_impurity_decrease
        ccp_alpha = hyper_params_regression_ccp_alpha
        hyper_parameters = product(
                criterion, 
                splitter, 
                max_depth, 
                min_samples_splits, 
                min_samples_leaf, 
                min_weight_fraction_leaf, 
                max_features, 
                random_state,
                max_leaf_nodes,
                min_impurity_decrease,
                ccp_alpha
            )
    else:
        raise NotImplementedError("Not Implemented")

    # TODO - use enum for this, and do classification hyperparams

    results = {}
    
    for p in hyper_parameters:
        criterion, splitter, max_depth, max_features, min_samples_leaf, min_samples_splits = p
        
        model = tree.DecisionTreeRegressor(criterion=criterion,
                                           splitter=splitter,
                                           max_depth=max_depth,
                                           max_features=max_features,
                                           min_samples_leaf=min_samples_leaf,
                                           min_samples_splits=min_samples_splits
                                           )
        
        
        # Use either simple_cross_validation or kfold below
        #result = simple_cross_validation(model=model, x=x, y=y, metrics={"mae": mean_absolute_error, "rve": explained_variance_score})
        result = kfold(model=model, x=x, y=y, k=5, metrics={"mae": mean_absolute_error, "rve": explained_variance_score})
        
        results[p] = {"model": model, "evaluation": result}
        
    # TODO: select the best model and return it
    return results