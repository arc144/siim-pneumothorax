import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def generate_bagging_splits(n_size, nfold, bagging_size_ratio=1, random_seed=143):
    '''Generate random bagging splits'''
    np.random.seed(random_seed)
    ref = range(n_size)
    out_size = int(bagging_size_ratio * n_size)

    splits = []
    for _ in range(nfold):
        t_index = np.random.randint(0, n_size, size=out_size)
        v_index = [j for j in ref if j not in t_index]
        splits.append((t_index, v_index))

    return splits

class LightGBM():
    '''Microsoft LightGBM class wrapper'''

    def __init__(self, objective='binary', metric='binary_logloss',
                 num_leaves=40, lr=0.005, bagging_fraction=0.7,
                 feature_fraction=0.6, bagging_frequency=6, device='gpu',
                 **kwargs):
        self.params = {
            "objective": objective,
            "metric": metric,
            "num_leaves": num_leaves,
            "learning_rate": lr,
            "bagging_fraction": bagging_fraction,
            "feature_fraction": feature_fraction,
            "bagging_freq": bagging_frequency,
            "bagging_seed": 42,
            "verbosity": -1,
            "seed": 42,
            "device": device,
            "gpu_platform_id": 0,
            "gpu_device_id": 0,
        }
        for key, value in kwargs.items():
            self.params[key] = value

        if self.params['metric'] in ['auc']:
            self.get_best_metric = max
        else:
            self.get_best_metric = min

    def fit(self, train_X, train_y, val_X, val_y, ES_rounds=100, steps=5000,
            verbose=150, return_oof_pred=True, **kwargs):
        # Train LGB model
        lgtrain = lgb.Dataset(train_X, label=train_y)
        lgval = lgb.Dataset(val_X, label=val_y)
        evals_result = {}
        self.model = lgb.train(self.params, lgtrain,
                               num_boost_round=steps,
                               valid_sets=[lgtrain, lgval],
                               early_stopping_rounds=ES_rounds,
                               verbose_eval=verbose,
                               evals_result=evals_result)
        if return_oof_pred:
            pred = self.predict(val_X, logloss=False)
            acc = np.mean(val_y == (pred > 0.5))
            print('Accuracy={:.4f}'.format(acc))
        else:
            pred = None
        return evals_result, pred

    def cv(self, X, Y, nfold=5, ES_rounds=100, steps=5000, random_seed=143,
           bootstrap=False, bagging_size_ratio=1, shuffle=True,
           return_oof_pred=False, splits=None):
        # Train LGB model using CV
        if splits is None:
            if bootstrap:
                splits = generate_bagging_splits(
                    X.shape[0], nfold,
                    bagging_size_ratio=bagging_size_ratio,
                    random_seed=random_seed)

            else:
                kf = KFold(n_splits=nfold, shuffle=shuffle,
                           random_state=random_seed)
                splits = kf.split(X, y=Y)

        oof_results = []
        acc_results = []
        for train_index, val_index in splits:
            x_train = X[train_index]
            y_train = Y[train_index]
            x_val = X[val_index]
            y_val = Y[val_index]

            _, oof_prediction = self.fit(train_X=x_train, train_y=y_train,
                                         val_X=x_val, val_y=y_val,
                                         ES_rounds=100,
                                         steps=10000,
                                         return_oof_pred=True)

            oof_results.extend(oof_prediction)
            acc_results.append(
                np.mean(y_val == (oof_prediction > 0.5))
            )

        acc_results = np.array(acc_results)
        print('Mean accuracy: {}, std {}'.format(
            acc_results.mean(), acc_results.std()))

        if return_oof_pred:
            return np.array(oof_results)

    def cv_predict(self, X, Y, test_X, nfold=5, ES_rounds=100, steps=5000,
                   random_seed=143, logloss=False,
                   bootstrap=False, bagging_size_ratio=1,
                   return_oof_pred=False, splits=None):
        '''Fit model using CV and predict test using the average
         of all folds'''
        if splits is None:
            if bootstrap:
                splits = generate_bagging_splits(
                    X.shape[0], nfold,
                    bagging_size_ratio=bagging_size_ratio,
                    random_seed=random_seed)

            else:
                kf = KFold(n_splits=nfold, shuffle=True,
                           random_state=random_seed)
                splits = kf.split(X, y=Y)

        kFold_results = []
        acc_results = []
        oof_results = []
        for i, (train_index, val_index) in enumerate(splits):
            x_train = X[train_index]
            y_train = Y[train_index]
            x_val = X[val_index]
            y_val = Y[val_index]

            evals_result, oof_prediction = self.fit(train_X=x_train, train_y=y_train,
                                                    val_X=x_val, val_y=y_val,
                                                    ES_rounds=100,
                                                    steps=10000,
                                                    return_oof_pred=return_oof_pred)
            if return_oof_pred:
                oof_results.extend(oof_prediction)
            if evals_result:
                kFold_results.append(
                    np.array(
                        self.get_best_metric(
                            evals_result['valid_1'][self.params['metric']])))
                acc_results.append(
                    np.mean(y_val == (oof_prediction > 0.5))
                )

            # Get predictions
            if not i:
                pred_y = self.predict(test_X, logloss=logloss)
            else:
                pred_y += self.predict(test_X, logloss=logloss)

        kFold_results = np.array(kFold_results)
        acc_results = np.array(acc_results)
        if kFold_results.size > 0:
            print('Mean val error: {}, std {} '.format(
                kFold_results.mean(), kFold_results.std()))
            print('Mean accuracy: {}, std {}'.format(
                acc_results.mean(), acc_results.std()))

        # Divide pred by the number of folds and return
        if return_oof_pred:
            return pred_y / nfold, np.array(oof_results)
        return pred_y / nfold

    def multi_seed_cv_predict(self, X, Y, test_X, nfold=5, ES_rounds=100,
                              steps=5000,
                              random_seed=[143, 135, 138], logloss=True,
                              bootstrap=False, bagging_size_ratio=1):
        '''Perform cv_predict for multiple seeds and avg them'''
        for i, seed in enumerate(random_seed):
            if not i:
                pred = self.cv_predict(X, Y, test_X, nfold=nfold,
                                       ES_rounds=ES_rounds, steps=steps,
                                       random_seed=seed, logloss=logloss,
                                       bootstrap=bootstrap,
                                       bagging_size_ratio=bagging_size_ratio)
            else:
                pred += self.cv_predict(X, Y, test_X, nfold=nfold,
                                        ES_rounds=ES_rounds, steps=steps,
                                        random_seed=seed, logloss=logloss,
                                        bootstrap=bootstrap,
                                        bagging_size_ratio=bagging_size_ratio)

        return pred / len(random_seed)

    def predict(self, test_X, logloss=False):
        '''Predict using a fitted model'''
        pred_y = self.model.predict(
            test_X, num_iteration=self.model.best_iteration)
        if logloss:
            pred_y = np.expm1(pred_y)
        return pred_y

    def fit_predict(self, train_X, train_y, test_X, val_X=None, val_y=None,
                    logloss=True, return_oof_pred=False, **kwargs):
        evals_result, oof_pred = self.fit(
            train_X, train_y, val_X, val_y, return_oof_pred=return_oof_pred)
        pred_y = self.predict(test_X, logloss)
        if return_oof_pred:
            return evals_result, pred_y, oof_pred
        else:
            return evals_result, pred_y

    def optmize_hyperparams(self, param_grid, X, Y,
                            cv=4, scoring='neg_mean_squared_error',
                            verbose=1):
        '''Use GridSearchCV to optimize models params'''
        params = self.params
        params['learning_rate'] = 0.05
        params['n_estimators'] = 1000
        gsearch1 = GridSearchCV(estimator=lgb.LGBMModel(**params),
                                param_grid=param_grid,
                                scoring=scoring,
                                n_jobs=1,
                                iid=False,
                                cv=4)
        gsearch1.fit(X, Y)
        scores = gsearch1.grid_scores_
        best_params = gsearch1.best_params_
        best_score = np.sqrt(-gsearch1.best_score_)
        if verbose > 0:
            if verbose > 1:
                print('Scores are: ', scores)
            print('Best params: ', best_params)
            print('Best score: ', best_score)