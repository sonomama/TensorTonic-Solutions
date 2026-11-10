import numpy as np

def newtons_method(X: list, y: list, lr_gd: float, n_iters: int) -> dict:
    """
    Returns gradient-descent and Newton MSE curves in a dictionary.
    """
    def grad(X, w, y):
        num_samples = np.shape(X)[0]
        return 2 / num_samples * np.transpose(X) @ (X @ w - y)

    def hessian(X):
        num_samples = np.shape(X)[0]
        return 2 / num_samples * np.transpose(X) @ X

    def loss(X, w, y):
        num_samples = np.shape(X)[0]
        return 1 / num_samples * np.square(np.linalg.norm(X @ w - y))
        
    X = np.array(X)
    num_samples = np.shape(X)[0]
    X = np.concat([X, np.ones((num_samples, 1))], axis=1)
    y = np.array(y).reshape(-1, 1)
    num_features = np.shape(X)[1]
    w_gd = np.zeros((num_features, 1))
    w_n = np.zeros((num_features, 1))

    # g = 2 / n * X^T (X w -y)
    # H = 2 / n X^T X

    loss_0 = loss(X, w_gd, y)
    losses = {"gd_losses": [loss_0], "newton_losses": [loss_0]}

    for _ in range(n_iters):
        w_gd -= lr_gd * grad(X, w_gd, y)

        hessian_vec = np.linalg.solve(hessian(X), grad(X, w_n, y))
        w_n -= hessian_vec

        losses["gd_losses"].append(loss(X, w_gd, y))
        losses["newton_losses"].append(loss(X, w_n, y))

    return losses
        

    
    
    