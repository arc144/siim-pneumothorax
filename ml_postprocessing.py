from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from Stacking.Models import LightGBM
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd


def preprocess(df, scaler=None):
    cat_vars = ['ViewPosition', 'PatientSex']
    for var in cat_vars:
        cat_list = 'var' + '_' + var
        cat_list = pd.get_dummies(df[var], prefix=var)
        df = df.join(cat_list)
        df.drop(var, axis=1, inplace=True)

    # scale_vars = ['SoftArea', 'Area', 'NInstances', 'PatientAge']
    # scale_vars = [x for x in df.columns if x not in ['ViewPosition', 'PatientSex', 'PatientAge', 'Target']]
    # if scaler is None:
    #     scaler = StandardScaler()
    #     # scaler = MinMaxScaler()
    #     df[scale_vars] = scaler.fit_transform(df[scale_vars])
    # else:
    #     df[scale_vars] = scaler.transform(df[scale_vars])
    #
    # return df, scaler
    return df, None


def read_csvs(paths, has_target=True):
    for i, p in enumerate(paths):
        if not i:
            df = pd.read_csv(p).set_index('PatientID')
        else:
            df = df.join(pd.read_csv(p).set_index('PatientID'), how='outer', rsuffix='_{}'.format(i))
            for c in ['PatientAge', 'PatientSex', 'ViewPosition']:
                df[c].update(df.pop('{}_{}'.format(c, i)))
            if has_target:
                df['Target'].update(df.pop('{}_{}'.format('Target', i)))
    return df.replace([np.inf, -np.inf], 0)


if __name__ == '__main__':
    # train_csv = [
    #     '/media/hdd/Kaggle/Pneumothorax/Saves/2ndLevel/oof/PANetDilatedResNet34_768_Fold0_oof_features.csv',
    #     '/media/hdd/Kaggle/Pneumothorax/Saves/2ndLevel/oof/PANetDilatedResNet34_768_Fold1_oof_features.csv',
    #     '/media/hdd/Kaggle/Pneumothorax/Saves/2ndLevel/oof/PANetResNet50_768_Fold0_oof_features.csv',
    #     '/media/hdd/Kaggle/Pneumothorax/Saves/2ndLevel/oof/PANetResNet50_768_Fold1_oof_features.csv',
    #     '/media/hdd/Kaggle/Pneumothorax/Saves/2ndLevel/oof/PANetResNet50_768_Fold2_oof_features.csv',
    #     '/media/hdd/Kaggle/Pneumothorax/Saves/2ndLevel/oof/PANetResNet50_768_Fold3_oof_features.csv',
    # ]

    train_csv = [
        '/media/hdd/Kaggle/Pneumothorax/Saves/2ndLevel/Holdout/PANetDilatedResNet34_768_Fold0_holdout_features.csv',
        '/media/hdd/Kaggle/Pneumothorax/Saves/2ndLevel/Holdout/PANetDilatedResNet34_768_Fold1_holdout_features.csv',
        '/media/hdd/Kaggle/Pneumothorax/Saves/2ndLevel/Holdout/PANetResNet50_768_Fold0_holdout_features.csv',
        '/media/hdd/Kaggle/Pneumothorax/Saves/2ndLevel/Holdout/PANetResNet50_768_Fold1_holdout_features.csv',
        '/media/hdd/Kaggle/Pneumothorax/Saves/2ndLevel/Holdout/PANetResNet50_768_Fold2_holdout_features.csv',
        '/media/hdd/Kaggle/Pneumothorax/Saves/2ndLevel/Holdout/PANetResNet50_768_Fold3_holdout_features.csv',
    ]

    test_csv = [
        '/media/hdd/Kaggle/Pneumothorax/Saves/2ndLevel/Test/PANetDilatedResNet34_768_Fold0_testset_features.csv',
        '/media/hdd/Kaggle/Pneumothorax/Saves/2ndLevel/Test/PANetDilatedResNet34_768_Fold1_testset_features.csv',
        '/media/hdd/Kaggle/Pneumothorax/Saves/2ndLevel/Test/PANetResNet50_768_Fold0_testset_features.csv',
        '/media/hdd/Kaggle/Pneumothorax/Saves/2ndLevel/Test/PANetResNet50_768_Fold1_testset_features.csv',
        '/media/hdd/Kaggle/Pneumothorax/Saves/2ndLevel/Test/PANetResNet50_768_Fold2_testset_features.csv',
        '/media/hdd/Kaggle/Pneumothorax/Saves/2ndLevel/Test/PANetResNet50_768_Fold3_testset_features.csv',
    ]
    sub_csv = '/media/hdd/Kaggle/Pneumothorax/Output/best_sub_no_noise_rmvl.csv'

    train_df = read_csvs(train_csv)
    train_y = train_df['Target'].values
    train_df = train_df.drop(['Target'], axis=1)

    # val_df = read_csvs(val_csv)
    # val_y = val_df['Target'].values
    # val_df = val_df.drop(['Target'], axis=1)

    test_df = read_csvs(test_csv, has_target=False)

    train_df, scaler = preprocess(train_df)
    # val_df, _ = preprocess(val_df, scaler)
    test_df, _ = preprocess(test_df, scaler)

    train_x = train_df.values
    # val_x = val_df.values
    test_x = test_df.values

    # Concat holdout and val
    # train_x = np.concatenate([train_x, val_x], axis=0)
    # train_y = np.concatenate([train_y, val_y], axis=0)

    LightGBM_params = dict(boosting='gbdt',
                           objective='binary',
                           num_leaves=15,
                           lr=0.01,
                           bagging_fraction=0.5,
                           max_depth=10,
                           max_bin=220,
                           feature_fraction=0.1,
                           bagging_freq=7,
                           min_data_in_leaf=12,
                           use_missing=True, zero_as_missing=False,
                           # min_split_gain=np.power(10, 0.1),
                           # min_child_weight=np.power(10., 0.7),
                           lambda_l1=np.power(10., -5),
                           lambda_l2=np.power(10., -5),
                           device='cpu', num_threads=11)

    fit_params = dict(nfold=20, ES_rounds=100,
                      steps=10000, random_seed=2019,
                      bootstrap=True, bagging_size_ratio=1.25)

    model = LightGBM(**LightGBM_params)
    # pred_test, oof_pred = model.fit(train_x, train_y, val_x, val_y,
    #                                 logloss=False, return_oof_pred=True,
    #                                 **fit_params)
    pred_test, oof_pred = model.cv_predict(train_x, train_y, test_x, return_oof_pred=True,
                                           **fit_params)
    # model = SVC(C=0.75, kernel='poly', degree=1, probability=True) # 915
    # model.fit(train_x, train_y)
    # print(model.score(val_x, val_y))
    # model0 = LinearSVC(C=0.75, loss='hinge') # 909
    # model2 = NuSVC(nu=0.3, kernel='rbf', degree=3, probability=True) # 913
    # model3 = LogisticRegression(max_iter=100, n_jobs=6)
    # model4 = GradientBoostingClassifier(learning_rate=0.05, max_features=0.5, max_depth=2, n_estimators=100,
    #                                    loss='exponential') # 0.
    #
    # n_splits = 10
    # # splits = generate_bagging_splits(train_x.shape[0], n_splits)
    # kf = KFold(n_splits=n_splits, shuffle=True,
    #            random_state=209)
    # splits = kf.split(train_x, y=train_y)
    # mean_acc = []
    # for i, (train_index, val_index) in enumerate(splits):
    #     x_train = train_x[train_index]
    #     y_train = train_y[train_index]
    #     x_val = train_x[val_index]
    #     y_val = train_y[val_index]
    #
    #     model = NuSVC(nu=0.3, kernel='rbf', degree=3, probability=True)
    #     model.fit(x_train, y_train)
    #     acc = model.score(x_val, y_val)
    #     mean_acc.append(acc)
    #     print('Accuracy: {:4f}'.format(acc))
    #
    #     if not i:
    #         pred_test = model.predict_proba(test_x) / n_splits
    #     else:
    #         pred_test = pred_test + model.predict_proba(test_x) / n_splits
    #
    # print('Mean acc {:.3f}'.format(np.mean(mean_acc)))
    # pred_test = np.argmax(pred_test, axis=1)

    pred_test = pred_test > 0.45
    print('{} non empty images'.format((pred_test == 1).sum()))

    ix = np.where(pred_test == 0)[0]
    sub = pd.read_csv(sub_csv)
    sub.loc[ix, 'EncodedPixels'] = '-1'
    sub.to_csv('/media/hdd/Kaggle/Pneumothorax/Output/fixed_sub.csv', index=False)
