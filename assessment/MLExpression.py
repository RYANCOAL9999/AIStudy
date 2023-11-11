import plotly.express as px

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from lib.feature.polynomialFeatures import polynomialFe
from lib.feature.linearExpressions import linearFe, ridgeFe, lassoFe, elasticNetFe

def showFigure(dataSet, key, dict):

    dataSet.isna().sum()
    dataSet['year'] = dataSet['date'].str.slice(0, 4) 
    dataSet['month'] = dataSet['date'].str.slice(4, 6) 
    dataSet['day'] = dataSet['date'].str.slice(6, 8) 
    dataSet = dataSet.drop('date', axis=1)
    dataSet = dataSet.drop('id', axis=1)
    dataSet.head(3)
    figureData = dict[key]

    if figureData == None:
        return None
    
    return px.histogram(
        dataSet, 
        x=key, 
        title=figureData["title"], 
        labels=figureData["labels"]
    )

def linearRegression(
        dataSet, 
        x_dataDrop_key, 
        y_dataDrop_key, 
        size, 
        random,
        **kwargs
    ):

    x_train, x_test, y_train, y_test = train_test_split(
        dataSet.drop(x_dataDrop_key, axis=1), 
        dataSet[y_dataDrop_key], 
        test_size=size, 
        random_state=random
    )

    mreg = linearFe(**kwargs).fit(x_train, y_train)

    return {
        "intercept": mreg.intercept_,
        "coef":mreg.coef_,
        "train_score":mreg.score(x_train, y_train),
        "test_score":mreg.score(x_test, y_test),
        "x_train":x_train,
        "y_train":y_train
    }


def polyRegression(
        dataSet, 
        x_dataDrop_key, 
        y_dataDrop_key, 
        size, 
        random,
        degree,
        **kwargs
    ):

    x_train, x_test, y_train, y_test = train_test_split(
        dataSet.drop(x_dataDrop_key, axis=1), 
        dataSet[y_dataDrop_key], 
        test_size=size, 
        random_state=random
    )

    poly_m = polynomialFe(
        degree=degree, 
        **{
            "interaction_only" : kwargs["interaction_only"] if "interaction_only" in kwargs else False,
            "include_bias" : kwargs["include_bias"] if "include_bias" in kwargs else True,
            "order" : kwargs["order"] if "order" in kwargs else "C"
        }
    )

    x_train_poly = poly_m.fit_transform(x_train)

    mreg_poly = linearFe(**{
        "fit_intercept": kwargs["fit_intercept"] if "fit_intercept" in kwargs else True,
        "copy_X" : kwargs["copy_X"] if "copy_X" in kwargs else True,
        "n_jobs" : kwargs["n_jobs"] if "n_jobs" in kwargs else True,
        "positive": kwargs["positive"] if "positive" in kwargs else True
    }).fit(x_train, y_train)

    x_test_poly = poly_m.fit_transform(x_test)

    return {
        "intercept": mreg_poly.intercept_,
        "coef":mreg_poly.coef_,
        "train_score":mreg_poly.score(x_train, y_train),
        "test_score":mreg_poly.score(x_test, y_test)
    }

def ridgeRegression(
        dataSet, 
        x_dataDrop_key, 
        y_dataDrop_key, 
        size, 
        random, 
        alpha=1, 
        **kwargs
    ):

    x_train, x_test, y_train, y_test = train_test_split(
        dataSet.drop(x_dataDrop_key, axis=1), 
        dataSet[y_dataDrop_key], 
        test_size=size, 
        random_state=random
    )

    ridge = ridgeFe(alpha=alpha, **kwargs).fit(x_train, y_train)

    return {
        "intercept": ridge.intercept_,
        "coef":ridge.coef_,
        "train_score":ridge.score(x_train, y_train),
        "test_score":ridge.score(x_test, y_test)
    }

def lassoRegression(
        dataSet, 
        x_dataDrop_key, 
        y_dataDrop_key, 
        size, 
        random, 
        alpha=1, 
        **kwargs
    ):

    x_train, x_test, y_train, y_test = train_test_split(
        dataSet.drop(x_dataDrop_key, axis=1), 
        dataSet[y_dataDrop_key], 
        test_size=size, 
        random_state=random
    )

    lasso = lassoFe(alpha=alpha, **kwargs).fit(x_train, y_train)

    return {
        "intercept": lasso.intercept_,
        "coef":lasso.coef_,
        "train_score":lasso.score(x_train, y_train),
        "test_score":lasso.score(x_test, y_test)
    }

def elasticRegression(
        dataSet, 
        x_dataDrop_key, 
        y_dataDrop_key, 
        size, 
        random, 
        alpha = 1, 
        **kwargs
    ):

    x_train, x_test, y_train, y_test = train_test_split(
        dataSet.drop(x_dataDrop_key, axis=1), 
        dataSet[y_dataDrop_key], 
        test_size=size, 
        random_state=random
    )

    elast = elasticNetFe(alpha=alpha, **kwargs).fit(x_train, y_train)

    return {
        "intercept": elast.intercept_,
        "coef":elast.coef_,
        "train_score":elast.score(x_train, y_train),
        "test_score":elast.score(x_test, y_test)
    }
def selection(
        dataSet,
        x_labels,
        y_labels
    ):

    x_train, x_test, y_train, y_test = train_test_split(
        dataSet[x_labels].to_numpy(), 
        dataSet[y_labels].transpose().to_numpy()[0]
    )

    model = linearFe().fit(x_train, y_train)

    y_pred = model.predict(x_test)

    x_test_transp = x_test.transpose()

    return {
        "model_score":model.score(x_train, y_train),
        "mean_absolute_error":mean_absolute_error(y_test, y_pred),
        "x_test_transp": x_test_transp,
        "y_test": y_test,
        "y_pred": y_pred
    }

    














