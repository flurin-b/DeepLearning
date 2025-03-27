"""Functions for training, evaluating and using a single neuron."""

import numpy as np

def sigmoid (a):
    return 1/(1+np.e**(-a))


def evaluate_neuron(x, w, b):
    """Calculate the output of a neuron with two input nodes.
    The sigmoid function is used as activation function.

    Args:
        x: input (numpy array of size [batch_size, 2])
        w: weights (numpy array of size [2])
        b: bias (numpy array of size [1])

    Returns:
        y, a: output and pre-activation (numpy arrays of size [batch_size])
    """

    assert x.ndim == 2
    assert x.shape[1] == 2
    assert w.shape == (2,)
    assert b.shape == (1,) or b.shape == ()
    
    a = np.dot(w,np.transpose(x)) + b
    y = sigmoid(a)
    
    return y, a


def derivative_of_sigmoid(a):
    """Calculate the derivative of the sigmoid function.

    Args:
        a: pre-activation of the neuron (numpy array of size [m, n])

    Returns:
        deriv: derivative (numpy array of size [m, n])
    """

    # d/de sigmoid(a) = sigmoid(a) * (1-sigmoid(a))$
    d_sigmoid = sigmoid(a) * (1.0-sigmoid(a))
    
    return d_sigmoid


def loss_function(x, t, w, b):
    """Calculate the loss function of the neuron.

    Args:
        x: input (numpy array of size [batch_size, 2])
        t: target, desired output (numpy array of size [batch_size])
        w: weights (numpy array of size [2])
        b: bias (numpy array of size [1])

    Returns:
        loss: the calculated loss (scalar)
    """

    assert x.ndim == 2
    assert x.shape[1] == 2
    assert t.ndim == 1
    assert t.shape[0] == x.shape[0]
    assert w.shape == (2,)
    assert b.shape == (1,) or b.shape == ()

    loss = 1.0/(2.0*x.shape[0]) * np.sum(np.square(evaluate_neuron(x, w, b)[0]-t))

    return loss


def update_weights(x, t, w, b, lr):
    """Update the weights and the bias by applying stochastic gradient descent.

    Args:
        x:  input (numpy array of size [batch_size, 2])
        t:  target, desired output (numpy array of size [batch_size])
        w:  weights (numpy array of size [2])
        b:  bias (numpy array of size [1])
        lr: learning rate

    Returns:
        w_new, b_new: updated weights and bias
    """

    assert x.ndim == 2
    assert x.shape[1] == 2
    assert t.ndim == 1
    assert t.shape[0] == x.shape[0]
    assert w.shape == (2,)
    assert b.shape == (1,) or b.shape == ()
    
    a = evaluate_neuron(x, w, b)[1]
    
    w_new = w - lr * np.sum(np.transpose(x)*(sigmoid(a) - t)*derivative_of_sigmoid(a), 1)/t.shape[0]
    b_new = b - lr * np.sum((sigmoid(a) - t)*derivative_of_sigmoid(a))/t.shape[0]
    
    return w_new, b_new


def evaluate_prediction(x, t, w, b):
    """Evaluate the prediction (predicted class) of the neuron.

    Args:
        x: input (numpy array of size [batch_size, 2])
        t: target, desired output (numpy array of size [batch_size])
        w: weights (numpy array of size [2])
        b: bias (numpy array of size [1])

    Returns:
        prediction: predicted output (numpy array of size [batch_size])
        accuracy: proportion of correct predictions
    """

    assert x.ndim == 2
    assert x.shape[1] == 2
    assert t.ndim == 1
    assert t.shape[0] == x.shape[0]
    assert w.shape == (2,)
    assert b.shape == (1,) or b.shape == ()

    
    prediction = evaluate_neuron(x, w, b)[0]
    prediction[prediction > 0.5] = 1
    prediction[prediction <= 0.5] = 0
    accuracy = len(prediction[prediction == t])/len(t)
    
    return prediction, accuracy


# Tests for the defined functions
if __name__ == "__main__":

    print("Start unit test for module neuron.py.")

    # test values
    x = np.array(
        [[1.56, 2.58], [-4.64, 2.43], [3.49, -1.08], [4.34, 1.55], [1.79, -3.29]]
    )
    w = np.array([2.06, -4.68])
    b = np.array(-2.23)
    t = np.array([1, 0, 0, 0, 1])
    y_target = np.array(
        [
            1.52517660984416e-05,
            8.73760411044216e-11,
            0.999955224291058,
            0.367350528619039,
            0.999999952121537,
        ]
    )
    a_target = np.array([-11.0908, -23.1608, 10.0138, -0.5436, 16.8546])
    loss_target = 0.213482635816409
    da_target = np.array(
        [
            1.52515334820725e-05,
            8.73760410967870e-11,
            4.47737040777816e-05,
            0.232404117742352,
            4.78784611854547e-08,
        ]
    )
    lr = 0.75
    w_new_target = np.array([2.00440180296548, -4.69983624753640])
    b_new_target = np.array([-2.24281049438565])
    prediction_target = np.array([0, 0, 1, 0, 1])
    accuracy_target = 0.6

    # test function evaluate_neuron
    y, a = evaluate_neuron(x, w, b)
    assert np.all(np.abs(a_target - a) < 10e-15)
    assert np.all(np.abs(y_target - y) < 10e-15)

    # test function derivative_of_sigmoid
    da = derivative_of_sigmoid(a_target)
    assert np.all(np.abs(da_target - da) < 10e-15)

    # test function loss_function
    loss = loss_function(x, t, w, b)
    assert np.abs(loss_target - loss) < 10e-15

    # test function update_weights
    w_new, b_new = update_weights(x, t, w, b, lr)
    assert np.all(np.abs(w_new_target - w_new) < 10e-15)
    assert np.all(np.abs(b_new_target - b_new) < 10e-15)

    # test function evaluate_prediction
    prediction, accuracy = evaluate_prediction(x, t, w, b)
    assert np.all(np.abs(prediction_target - prediction) < 10e-15)
    assert np.abs(accuracy_target - accuracy) < 10e-15

    print("Unit test was successful.")
