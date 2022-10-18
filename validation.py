import numpy as np
from sklearn import tree
from itertools import product

def simple_cross_validation(*, model, x, y, train_partition=(0, 0.8), metrics={"mae": mean_absolute_error}):
    n, m = x.shape
    
    start = int(train_partition[0] * n)
    end = int(train_partition[1] * n)
    
    x_train = x[start:end, :]
    y_train = y[start:end]
    x_test = np.concatenate((x[:start, :], x[end:, :]))
    y_test = np.concatenate((y[:start], y[end:]))
    
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    return {metric_name: metric_f(y_test, y_pred) for (metric_name, metric_f) in metrics.items()}


def kfold(*, model, x, y, k=5, metrics={"mae": mean_absolute_error}):
    n, m = x.shape
    results = []
    
    for i in range(k):
        start = i * (1/k)
        end = start + (1/k)
        if (i + 1) == k:
            end = 1.0
            
        result = simple_cross_validation(model=model, x=x, y=y, train_partition=(start, end), metrics=metrics)
        
        results.append(result)
    
    return results


def model_selection(
    model_type: str, # Classification or Regression
    x_train: np.ndarray, x_dev: np.ndarray, x_test: np.nd_array, 
    y_train: np.ndarray, y_dev: np.ndarray, y_test: np.nd_array,
    hyper_params_regression = {
            "criterions": ["squared_error"],
            "splitters": ["best"],
            "max_depths": None,
            "min_samples_splits": range(2,30,1),
            "min_samples_leafs": range(1,30,1),
            #"min_weight_fraction_leaf": [0.0],
            "max_features": ["auto", "sqrt", "log2"]
            #random_state: [int] [None] - none is default, it is int
            #max_leaf_nodes: [int] [None] - none is default, it is int
            #min_impurity_decrease: [0.0]
            #ccp_alpha: [0.0]
        }):
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
    
    # Hyperparameters
    if model_type == "Regression":
        criterion = hyper_params_regression["criterions"]
        splitter = hyper_params_regression["splitters"]
        max_depth = hyper_params_regression["max_depths"]
        max_features = hyper_params_regression["max_features"]
        min_samples_leaf = hyper_params_regression["min_samples_leaf"]
        min_samples_splits = hyper_params_regression["min_samples_splits"]
        hyper_parameters = product(criterion, splitter, max_depth, max_features, min_samples_leaf, min_samples_splits)
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