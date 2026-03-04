import math

# ==========================================================
# 1️⃣ STATISTICAL FUNCTIONS
# ==========================================================

def mean(data):
    """
    Mean (μ) = Σx / n
    """
    return sum(data) / len(data)


def median(data):
    """
    Median:
    Sort data
    If n odd → middle
    If n even → avg of two middle
    """
    sorted_data = sorted(data)
    n = len(sorted_data)
    mid = n // 2

    if n % 2 == 0:
        return (sorted_data[mid - 1] + sorted_data[mid]) / 2
    return sorted_data[mid]


def variance_population(data):
    """
    Population Variance:
    σ² = Σ(x - μ)² / n
    """
    m = mean(data)
    return sum((x - m) ** 2 for x in data) / len(data)


def variance_sample(data):
    """
    Sample Variance:
    s² = Σ(x - x̄)² / (n - 1)
    """
    m = mean(data)
    return sum((x - m) ** 2 for x in data) / (len(data) - 1)


def standard_deviation_population(data):
    """
    σ = √(Population Variance)
    """
    return math.sqrt(variance_population(data))


def standard_deviation_sample(data):
    """
    s = √(Sample Variance)
    """
    return math.sqrt(variance_sample(data))


def z_score(x, data):
    """
    Z = (x - μ) / σ
    """
    m = mean(data)
    sd = standard_deviation_population(data)
    return (x - m) / sd


def covariance(x, y):
    """
    Cov(X,Y) = Σ[(xi - x̄)(yi - ȳ)] / n
    """
    mx = mean(x)
    my = mean(y)
    n = len(x)
    return sum((x[i] - mx) * (y[i] - my) for i in range(n)) / n


def correlation(x, y):
    """
    Correlation:
    r = Cov(X,Y) / (σx * σy)
    """
    return covariance(x, y) / (
        standard_deviation_population(x) *
        standard_deviation_population(y)
    )


# ==========================================================
# 2️⃣ LINEAR ALGEBRA FUNCTIONS
# ==========================================================

def dot_product(v1, v2):
    """
    A · B = Σ(ai * bi)
    """
    return sum(a * b for a, b in zip(v1, v2))


def matrix_add(A, B):
    """
    C[i][j] = A[i][j] + B[i][j]
    """
    return [
        [A[i][j] + B[i][j] for j in range(len(A[0]))]
        for i in range(len(A))
    ]


def matrix_multiply(A, B):
    """
    C[i][j] = Σ A[i][k] * B[k][j]
    """
    result = [[0] * len(B[0]) for _ in range(len(A))]

    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]

    return result


def determinant_2x2(matrix):
    """
    |A| = ad - bc
    """
    return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]


def inverse_2x2(matrix):
    """
    A⁻¹ = 1/det * [d  -b
                   -c  a]
    """
    det = determinant_2x2(matrix)
    if det == 0:
        raise ValueError("Matrix not invertible")

    return [
        [ matrix[1][1]/det, -matrix[0][1]/det ],
        [-matrix[1][0]/det,  matrix[0][0]/det ]
    ]


# ==========================================================
# 3️⃣ SIMPLE LINEAR REGRESSION (FROM SCRATCH)
# ==========================================================

def linear_regression(x, y):
    """
    Simple Linear Regression:
    y = b0 + b1x

    b1 = Cov(X,Y) / Var(X)
    b0 = ȳ - b1x̄
    """

    b1 = covariance(x, y) / variance_population(x)
    b0 = mean(y) - b1 * mean(x)

    return b0, b1


def predict(x_value, b0, b1):
    """
    ŷ = b0 + b1x
    """
    return b0 + b1 * x_value


# ==========================================================
# 4️⃣ TESTING SECTION
# ==========================================================

if __name__ == "__main__":

    data = [10, 20, 30, 40, 50]

    print("Mean:", mean(data))
    print("Median:", median(data))
    print("Variance (Pop):", variance_population(data))
    print("Std Dev (Pop):", standard_deviation_population(data))
    print("Z-score of 30:", z_score(30, data))

    x = [1, 2, 3, 4, 5]
    y = [2, 4, 5, 4, 5]

    print("Covariance:", covariance(x, y))
    print("Correlation:", correlation(x, y))

    b0, b1 = linear_regression(x, y)
    print("Regression Equation: y =", b0, "+", b1, "x")
    print("Prediction for x=6:", predict(6, b0, b1))

    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]

    print("Matrix Add:", matrix_add(A, B))
    print("Matrix Multiply:", matrix_multiply(A, B))
    print("Determinant:", determinant_2x2(A))
    print("Inverse:", inverse_2x2(A))