# https://scikit-learn.org/stable/modules/linear_model.html#
# https://colab.research.google.com/?utm_source=scs-index
# https://medium.com/analytics-vidhya/clustering-on-iris-dataset-in-python-using-k-means-4735b181affe
# https://colab.research.google.com/drive/14cdBB8F3ZyJPeTnUq7p9CFdO5j5xK8bo#scrollTo=JzfNBbmwHT8t
# https://intellipaat.com/blog/tutorial/python-tutorial/python-pandas-tutorial/

##------------------------------------------------------------------------------------------------------#
import numpy as np
from lib.feature.svmExpressions import LinearSVCFe, LinearSVRFe, SVCFe
from lib.feature.linearExpressions import bayesianRidgeFe, tweedieRegressorFe
from lib.feature.linearExpressions import lassoFe, lassoLarsFe
from lib.feature.nearestNeighbors import nearestNeighborsFe, kdTreeFe, nearestCentroidFe
from lib.feature.polynomialFeatures import polynomialFe, polynomialTransformFe
from lib.feature.linearExpressions import linearFe, ridgeFe, ridgeCVFe, sgdClassifierFe

Regression = linearFe(
    [[0, 0], [1, 1], [2, 2]], 
    [0, 1, 2]
)

print(Regression.coef_, "\n")

Ridge = ridgeFe(
    0.5, 
    [[0, 0], [1, 1], [2, 2]], 
    [0, 1, 2]
)

print(Ridge.coef_, "\n")

print(Ridge.intercept_, "\n")

RidgeCV = ridgeCVFe(
    np.logspace(
        -6, 
        6, 
        13
    ),
    [[0, 0], [0, 0], [1, 1]], 
    [0, .1, 1]
)

print(RidgeCV.alpha_, "\n")

Lasso = lassoFe(
    0.1,
    [[0, 0], [1, 1]], 
    [0, 1]
)

print(
    Lasso.predict(
        [[1, 1]]
    )
    , "\n"
)

LassoLarsLasso = lassoLarsFe(
    0.1, 
    [[0, 0], [1, 1]], 
    [0, 1]
)

print(LassoLarsLasso.coef_, "\n")

SianRidge = bayesianRidgeFe(
    [[0., 0.], [1., 1.], [2., 2.], [3., 3.]], 
    [0., 1., 2., 3.]
)

print(
    SianRidge.predict(
        [[1, 0.]]
    )
    , "\n"
)

print(SianRidge.coef_, "\n")

Tweedie = tweedieRegressorFe(
    [[0, 0], [0, 1], [2, 2]], 
    [0, 1, 2],
    power=1, 
    alpha=0.5, 
    link='log'
)

print(Tweedie.coef_, "\n")

print(Tweedie.intercept_, "\n")

Polynomial = polynomialFe(
    np.arange(6).reshape(3, 2),
    2
)

print(Polynomial, "\n")

PolynomialFitTransform = polynomialTransformFe(
    np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
    None,
    2,
    interaction_only = False,
    include_bias = True,
    order = "C", 
    fit_intercept=False, 
    max_iter=10, 
    tol=None,
    shuffle=False
)

print(PolynomialFitTransform["predict"], "\n")

print(PolynomialFitTransform["score"], "\n")

PredictionsWithTwoPoints = SVCFe(
    [[0, 0], [1, 1]],
    [0, 1]
)

print(
    PredictionsWithTwoPoints.predict(
        [[2., 2.]]
    )
    , "\n"
)

PredictionsWithOVO = SVCFe(
    [[0], [1], [2], [3]],
    [0, 1, 2, 3],
    decision_function_shape='ovo'
)

print(
    PredictionsWithOVO.decision_function(
        [[1]]
    )
    , "\n"
)

PredictionsWithOVO.decision_function_shape = "ovr"

print(
    PredictionsWithOVO.decision_function(
        [[1]]
    )
    , "\n"
)

LinearSVCExpression = LinearSVCFe(
    [[0], [1], [2], [3]],
    [0, 1, 2, 3],
    dual="auto"
)

print(
    LinearSVCExpression.decision_function(
        [[1]]
    ).shape[1]
    , "\n"
)

LinearSVRExpression = LinearSVRFe(
    [[0, 0], [2, 2]],
    [0.5, 2.5]
)

print(
    LinearSVRExpression.predict(
        [[1, 1]]
    )
    , "\n"
)

SGDClassifierExpression = sgdClassifierFe(
    [[0., 0.], [1., 1.]],
    [0, 1],
    loss="log_loss", 
    max_iter=5
)

print(
    SGDClassifierExpression.predict(
        [[2., 2.]]
    )
    , "\n"
)

print(SGDClassifierExpression.coef_, "\n")

print(SGDClassifierExpression.intercept_, "\n")

print(
    SGDClassifierExpression.decision_function(
        [[2., 2.]]
    )
    , "\n"
)

print(
    SGDClassifierExpression.predict_proba(
        [[1., 1.]]
    )
    , "\n"
)

NearestExpression = nearestNeighborsFe(
    np.asarray([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]),
    n_neighbors=2, 
    algorithm='ball_tree'
)

print(NearestExpression["indices"], "\n")

print(NearestExpression["distances"], "\n")

print(NearestExpression["arrayDisplay"], "\n")

KDTExpression = kdTreeFe(
    np.asarray([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]),
    2, 
    return_distance=False,
    leaf_size=30, 
    metric='euclidean'
)

print(KDTExpression, "\n")

NearestCentroidExpression = nearestCentroidFe(
    np.asarray([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]),
    np.asarray([1, 1, 1, 2, 2, 2])
)

print(
    NearestCentroidExpression.predict(
        [[-0.8, -1]]
    )
)


