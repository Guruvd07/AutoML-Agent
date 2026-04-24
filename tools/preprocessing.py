import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess(
        df,
        target=None,
        scaler=None,
        fit_scaler=True,
        training_columns=None
):
    df = df.copy()

    # ==================================
    # TARGET ENCODING
    # ==================================
    if target is not None and target in df.columns:

        if df[target].dtype == "object":

            vals = set(df[target].dropna().unique())

            if vals == {"Y","N"}:
                df[target] = df[target].map({
                    "Y":1,
                    "N":0
                })

    # ==================================
    # HANDLE MISSING VALUES
    # ==================================
    for col in df.columns:

        if df[col].dtype in ["int64","float64"]:
            df[col] = df[col].fillna(
                df[col].median()
            )

        else:
            mode_val = df[col].mode()

            df[col] = df[col].fillna(
                mode_val[0] if not mode_val.empty
                else "Unknown"
            )

    # ==================================
    # ONE HOT ENCODING
    # ==================================
    cat_cols = df.select_dtypes(
        include=["object"]
    ).columns.tolist()

    if target in cat_cols:
        cat_cols.remove(target)

    df = pd.get_dummies(
        df,
        columns=cat_cols,
        drop_first=True
    )

    # ==================================
    # SPLIT X / y
    # ==================================
    if target is not None and target in df.columns:

        X = df.drop(columns=[target])
        y = df[target]

    else:
        X = df
        y = None


    # ==================================
    # ALIGN COLUMNS FOR PREDICTION
    # ==================================
    if training_columns is not None:

        X = X.reindex(
            columns=training_columns,
            fill_value=0
        )


    # ==================================
    # SCALING
    # ==================================
    if fit_scaler:

        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X)

    else:

        X_scaled = scaler.transform(X)


    X_scaled = pd.DataFrame(
        X_scaled,
        columns=X.columns,
        index=X.index
    )


    # ==================================
    # RETURN TRAINING MODE
    # ==================================
    if fit_scaler:

        if y is not None:
            processed = pd.concat(
                [X_scaled,y],
                axis=1
            )
        else:
            processed = X_scaled

        return (
            processed,
            scaler,
            list(X.columns)
        )


    # ==================================
    # RETURN PREDICTION MODE
    # ==================================
    return X_scaled