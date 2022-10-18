import numpy as np

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
    x_train: np.ndarray, x_dev: np.ndarray, x_test: np.nd_array, 
    y_train: np.ndarray, y_dev: np.ndarray, y_test: np.nd_array, 
    hyper_params = {
            "criterions": ["squared_error", "friedman_mse", "absolute_error","poisson"],
            "splitters": ["best", "random"],
            "max_depths": range(10,100,10),
            "max_features": ["auto", "sqrt", "log2"],
            "min_samples_leaf": range(1,30,1),
            "min_samples_splits": range(2,30,1)
        }):
    
    # Split into train and test
    n, m = x.shape
    partition = int(0.8 * n)
    x_train = x[:partition, :]
    x_test = x[partition:, :]
    y_train = y[:partition]
    y_test = y[partition:]
    
    # Hyperparameters
    criterions = ["squared_error"]
    splitters = ["best"]
    max_depths = [3, 4, 12, None]
    hyper_parameters = product(criterions, splitters, max_depths)
    
    results = {}
    
    for p in hyper_parameters:
        criterion, splitter, max_depth = p
        
        model = tree.DecisionTreeRegressor(criterion=criterion,
                                           splitter=splitter,
                                           max_depth=max_depth)
        
        
        # Use either simple_cross_validation or kfold below
        #result = simple_cross_validation(model=model, x=x, y=y, metrics={"mae": mean_absolute_error, "rve": explained_variance_score})
        result = kfold(model=model, x=x, y=y, k=5, metrics={"mae": mean_absolute_error, "rve": explained_variance_score})
        
        results[p] = {"model": model, "evaluation": result}
        
    # TODO: select the best model and return it
    return results