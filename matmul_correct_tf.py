import time
import numpy
import tensorflow as tf

def genereate_matrix(n=1024):
    numbers = numpy.arange(n * n, dtype=numpy.uint64)
    X = numbers.reshape((n, n))
    return X


def genereate_random_matrix(n=1024):
    rng = numpy.random.default_rng()
    X = rng.random((n, n))
    return X


def check_result(ground_truth: numpy.ndarray, res: tf.Tensor):
    assert isinstance(ground_truth, numpy.ndarray)
    assert isinstance(res, numpy.ndarray)
    return numpy.sum(numpy.abs(ground_truth - res) / ground_truth) / ground_truth.size


def calc(matrix, ground_truth, device, dtype):
    if device == "numpy":
        calc_numpy(matrix=matrix, ground_truth=ground_truth, dtype=dtype)
    else:        
        calc_tf(matrix=matrix, ground_truth=ground_truth, device=device, dtype=dtype)


def calc_tf(matrix, ground_truth, device, dtype):
    (X, Y) = matrix
    X_fp = X.astype(dtype=dtype)
    Y_fp = Y.astype(dtype=dtype)
    with tf.device(device):
        X_t_fp = tf.convert_to_tensor(X_fp, dtype=dtype)
        Y_t_fp = tf.convert_to_tensor(Y_fp, dtype=dtype)

        begin = time.time()
        res_t_fp = tf.matmul(X_t_fp, Y_t_fp).numpy()
        end = time.time()
    error = check_result(ground_truth, res_t_fp)
    metric, metric_name = calc_metric(end - begin, X.shape[0])
    print(f"{str(device): <16} {dtype}, {X.shape}: {metric:>6.1f} {metric_name}, error: {error:5.2E}")
    return end - begin


def calc_numpy(matrix, ground_truth, dtype):
    (X, Y) = matrix
    X_fp = X.astype(dtype=dtype)
    Y_fp = Y.astype(dtype=dtype)
    begin = time.time()
    res_fp = numpy.matmul(X_fp, Y_fp)
    end = time.time()
    error = check_result(ground_truth, res_fp)
    metric, metric_name = calc_metric(end - begin, X.shape[0])
    print(f"{'numpy': <16} {dtype}, {X.shape}: {metric:>6.1f} {metric_name}, error: {error:5.2E}")


def calc_metric(time, n):
    N = float(n)
    FLOP = 2 * N * N * (N - 1)
    GFLOPs = FLOP / time / 1e9
    return GFLOPs, "GFLOP/s"


def run(n_int=1024, n_float=1024, r=10):
    physical_devices = tf.config.list_logical_devices()
    physical_devices = list(map(lambda x: x.name, physical_devices))
    devices = ["numpy"] + physical_devices
    dtypes = [numpy.float32, numpy.float64]

    print("devices: ", devices)
    print("dtypes: ", dtypes)

    print("calculating with ground truth uint64")
    for i in range(r):
        X = genereate_matrix(n_int)
        Y = genereate_matrix(n_int)
        ground_truth = numpy.matmul(X, Y)  # integer MatMul does not have floating point errors


        for device in devices:
            for dtype in dtypes:
                calc((X, Y), ground_truth=ground_truth, device=device, dtype=dtype)

    print("calculating with ground truth random float64")
    for i in range(r):
        X = genereate_random_matrix(n_float)
        Y = genereate_random_matrix(n_float)
        ground_truth = numpy.matmul(X, Y)  # integer MatMul does not have floating point errors

        for device in devices:
            for dtype in dtypes:
                calc((X, Y), ground_truth=ground_truth, device=device, dtype=dtype)


if __name__ == "__main__":
    run(n_int=1024, n_float=8192, r=5)
