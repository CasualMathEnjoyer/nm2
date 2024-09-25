import numpy as np

class SparseMatrix:
    def __init__(self, sizex, sizey):
        self.sizex = sizex
        self.sizey = sizey
        self.shape = (sizex, sizey)
        self.positions = [{} for _ in range(sizex)]

    def __getitem__(self, touplekey):
        if isinstance(touplekey, int):
            if 0 <= touplekey < self.sizex:
                return self.positions[touplekey]
            else:
                raise IndexError
        keyx, keyy = touplekey
        if keyx >= self.sizex or keyy >= self.sizey:
            raise IndexError
        return self.positions[keyx].get(keyy, 0)

    def __setitem__(self, touplekey, value):
        keyx, keyy = touplekey
        if keyx >= self.sizex or keyy >= self.sizey:
            raise IndexError
        if value == 0:
            if keyy in self.positions[keyx]:
                del self.positions[keyx][keyy]
        else:
            self.positions[keyx][keyy] = value

    def __add__(self, other):
        if isinstance(other, SparseMatrix):
            if self.sizex != other.sizex or self.sizey != other.sizey:
                raise IndexError("Matrices do not have the same size")
            result = SparseMatrix(self.sizex, self.sizey)
            for i in range(self.sizex):
                for j in self.positions[i]:
                    result[i, j] = self[i, j] + other[i, j]
                for j in other.positions[i]:
                    if j not in self.positions[i]:
                        result[i, j] = other[i, j]
            return result
        elif isinstance(other, (float, int)):
            result = SparseMatrix(self.sizex, self.sizey)
            for i in range(self.sizex):
                for j in self.positions[i]:
                    result[i, j] = self[i, j] + other
            return result

    def __len__(self):
        return self.sizey

    def __repr__(self):
        return f"SparseMatrix({self.sizex}, {self.sizey}, {self.positions})"

    def __matmul__(self, other):
        if isinstance(other, SparseMatrix):
            if self.sizey != other.sizex:
                raise IndexError("Matrix dimensions do not align for multiplication")
            result = SparseMatrix(self.sizex, other.sizey)
            for i in range(self.sizex):
                for j in self.positions[i]:
                    if j in other.positions:
                        for k in other.positions[j]:
                            result[i, k] += self[i, j] * other[j, k]
            return result
        elif isinstance(other, np.ndarray) and other.ndim == 1:
            if self.sizey != other.size:
                raise IndexError("Matrix and vector dimensions do not align for multiplication")
            result = np.zeros(self.sizex)
            for i in range(self.sizex):
                for j in self.positions[i]:
                    result[i] += self[i, j] * other[j]
            return result
        else:
            raise TypeError("Unsupported type for multiplication")

if __name__ == "__main__":
    # Example usage
    matrix = SparseMatrix(sizex=5, sizey=5)
    matrix[0, 0] = 1
    matrix[1, 3] = 2
    print(matrix[0, 0])  # Output: 1
    print(matrix[1, 3])  # Output: 2
    print(matrix[2, 2])  # Output: 0
    print(matrix)

    # Sparse matrix multiplication
    other_matrix = SparseMatrix(sizex=5, sizey=5)
    other_matrix[0, 0] = 3
    other_matrix[3, 1] = 4
    result_matrix = matrix @ other_matrix
    print(result_matrix)

    # Sparse matrix and 1D numpy array multiplication
    vector = np.array([1, 2, 3, 4, 5])
    result_vector = matrix @ vector
    print(result_vector)
