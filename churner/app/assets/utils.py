import os
import pickle

import sklearn
import numpy as np


def formater(df, list_variables, new_type):
    df[list_variables] = df[list_variables].astype(new_type)
    return df


def nan_replacer(df, list_variables, list_symbols):
    for symbol in list_symbols:
        df[list_variables] = df[list_variables].replace(symbol, np.nan)
    return df


def encoder(df, list_categorical, label, savepath=".", option=None):
    option = "train" if option is None else option

    if option == "inference":
        list_to_encode = list_categorical
    else:
        list_to_encode = list_categorical + [label]

    for var in list_to_encode:
        name = "{}.pkl".format(var.lower())
        var_savepath = os.path.join(savepath, name)
        if option == "train":
            # instancier un label encodeur
            le = sklearn.preprocessing.LabelEncoder()
            # Entrainer le label encodeur
            le.fit(df[var])
            # Sauvegarder le label encodeur
            with open(var_savepath, "wb") as file:
                pickle.dump(le, file)
        else:
            # Charger le label encodeur
            with open(var_savepath, "rb") as file:
                le = pickle.load(file)
        # Transformer la colonne avec le label encodeur entrainé préalablement
        df[var] = le.transform(df[var])

    print("======= L'encodage des variables catégorielles est terminée ! =======")
    return df


def imputer_median(X, var_names, col="TotalCharges", savepath="."):
    id_ToCh = var_names.index(col)
    col_values = X[:, id_ToCh]
    name = "{}_median.pkl".format(col)
    var_savepath = os.path.join(savepath, name)

    # Compute the median
    median = np.nanmedian(col_values)
    # Save the median as an artifact
    with open(var_savepath, "wb") as file:
        pickle.dump(median, file)

    return X


def preprocessor(df, config, option_output=None, option_train=None):
    df = df.copy()
    option_output = "all" if option_output is None else option_output
    option_train = "train" if option_train is None else option_train
    if option_train == "train":
        # Suppression des doublons
        df = df.drop_duplicates()
    # Remplacement des valeurs symbols par nan
    df = nan_replacer(
        df, list_variables=config["data"]["numerical"], list_symbols=[" "]
    )
    # Conversion des variables numériques au bon format
    df = formater(df, list_variables=config["data"]["numerical"], new_type="float32")
    # Conversion des variables catégorielles au bon format
    df = formater(df, list_variables=config["data"]["categorical"], new_type="object")
    # Encoder les variables catégorielles
    df = encoder(
        df,
        list_categorical=config["data"]["categorical"],
        label=config["data"]["label"],
        savepath=config["model"]["savepath"],
        option=option_train,
    )
    # Separate label and other variables
    df = df.reset_index(drop=True)
    var_names = config["data"]["numerical"] + config["data"]["categorical"]
    X = df[var_names].values
    if option_train == "train":
        # Impute the median for TotalCharges
        X = imputer_median(
            X, var_names, col="TotalCharges", savepath=config["model"]["savepath"]
        )

    if option_output == "all":
        y = df[config["data"]["label"]].values
        output = X, y
    if option_output == "inference":
        output = X

    return output
