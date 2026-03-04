import math

# ==========================================
# STATISTICAL FUNCTIONS
# ==========================================

def mean(data):
    """Mean = Σx / n"""
    return sum(data) / len(data)


def median(data):
    """Median calculation"""
    sorted_data = sorted(data)
    n = len(sorted_data)
    mid = n // 2

    if n % 2 == 0:
        return (sorted_data[mid - 1] + sorted_data[mid]) / 2
    else:
        return sorted_data[mid]


def variance_population(data):
    """Population Variance = Σ(x - μ)² / n"""
    m = mean(data)
    return sum((x - m) ** 2 for x in data) / len(data)


def variance_sample(data):
    """Sample Variance = Σ(x - x̄)² / (n-1)"""
    m = mean(data)
    return sum((x - m) ** 2 for x in data) / (len(data) - 1)


def standard_deviation_population(data):
    """Population Std Dev = √variance"""
    return math.sqrt(variance_population(data))


def standard_deviation_sample(data):
    """Sample Std Dev = √variance"""
    return math.sqrt(variance_sample(data))


def z_score(x, data):
    """Z = (x - μ) / σ"""
    m = mean(data)
    sd = standard_deviation_population(data)
    return (x - m) / sd


def data_range(data):
    """Range = max - min"""
    return max(data) - min(data)


# ==========================================
# LINEAR ALGEBRA FUNCTIONS
# ==========================================

def dot_product(v1, v2):
    """Dot Product = Σ(ai * bi)"""
    if len(v1) != len(v2):
        raise ValueError("Vectors must be same length")
    return sum(a * b for a, b in zip(v1, v2))


def matrix_add(A, B):
    """Matrix Addition"""
    result = []
    for i in range(len(A)):
        row = []
        for j in range(len(A[0])):
            row.append(A[i][j] + B[i][j])
        result.append(row)
    return result


def matrix_multiply(A, B):
    """Matrix Multiplication"""
    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]

    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]

    return result


# ==========================================
# TESTING SECTION
# ==========================================

if __name__ == "__main__":

    data = [10, 20, 30, 40, 50]

    print("Mean:", mean(data))
    print("Median:", median(data))
    print("Population Variance:", variance_population(data))
    print("Sample Variance:", variance_sample(data))
    print("Population Std Dev:", standard_deviation_population(data))
    print("Sample Std Dev:", standard_deviation_sample(data))
    print("Z-Score of 30:", z_score(30, data))
    print("Range:", data_range(data))

    v1 = [1, 2, 3]
    v2 = [4, 5, 6]

    print("Dot Product:", dot_product(v1, v2))

    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]

    print("Matrix Addition:", matrix_add(A, B))
    print("Matrix Multiplication:", matrix_multiply(A, B))