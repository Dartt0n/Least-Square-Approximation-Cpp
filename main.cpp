// Anton Kudryavtsev
// a.kudryavtsev@innopolis.university
#include <iostream>
#include <functional>
#include <cmath>


constexpr void free(std::initializer_list<void*> pointers) {
    for (auto p : pointers) {
        delete p;
    }
}


// matrix.hpp
template<typename T>
concept required_accuracy = std::is_arithmetic_v<T>;

class DimensionalException;

class SingularMatrixException;

template<required_accuracy T>
class Matrix;

template<required_accuracy T>
class SquareMatrix;

template<required_accuracy T>
class ColumnVector;

template<required_accuracy T>
class EliminationMatrix;

template<required_accuracy T>
class IdentityMatrix;

template<required_accuracy T>
class PermutationMatrix;

template<required_accuracy T>
class ColumnVector;

// matrix.cpp
class DimensionalException : public std::exception {
public:
    const char *what() {
        return "Error: the dimensional problem occurred";
    }
};

class SingularMatrixException : public std::exception {
public:
    const char *what() {
        return "Error: the matrix is singular";
    }
};

template<required_accuracy T>
class Matrix {
protected:
    size_t rows{}, cols{};
    T *raw_data;

    [[nodiscard]] bool matchSize(const Matrix<T> *matrix) const {
        return this->cols == matrix->cols && this->rows == matrix->rows;
    }

    [[nodiscard]] T error_threshold(T value) const {
        return std::abs(value) <= ERROR_THRESHOLD ? 0.00 : value;
    }

public:
    constexpr static const double ERROR_THRESHOLD = 10e-10;

    Matrix(size_t rows, size_t cols) {
        this->rows = rows;
        this->cols = cols;
        raw_data = new T[rows * cols];
    }

    Matrix(size_t rows, size_t cols, std::function<T(size_t, size_t)> func) : Matrix(rows, cols) {
        for (size_t i = 0; i < this->rows; ++i) {
            for (int j = 0; j < this->cols; ++j) {
                this->Put(i, j, func(i, j));
            }
        }
    }

    virtual ~Matrix() {
        delete[] this->raw_data;
    }

    Matrix<T> &operator=(const Matrix<T> &matrix) {
        if (!this->matchSize(matrix)) {
            throw DimensionalException();
        }

        if (this == matrix) {
            return *this;
        }

        this->Map([&matrix](auto i, auto j, auto old) { return matrix.Get(i, j); });

        return *this;
    }

    [[nodiscard]] virtual T Get(size_t row, size_t col) const {
        return this->raw_data[row * this->cols + col];
    }

    [[nodiscard]] size_t Rows() const {
        return this->rows;
    }

    [[nodiscard]] size_t Cols() const {
        return this->cols;
    }

    virtual void Put(size_t row, size_t col, T value) {
        this->raw_data[row * this->cols + col] = value;
    }

    void Map(std::function<T(size_t, size_t, T)> func) {
        for (int i = 0; i < this->rows; ++i) {
            for (int j = 0; j < this->cols; ++j) {
                this->Put(i, j, func(i, j, this->raw_data[i * this->cols + j]));
            }
        }
    }

    Matrix<T> *operator+(const Matrix<T> *matrix) const {
        if (!this->matchSize(matrix)) {
            throw DimensionalException();
        }

        return new Matrix<T>(this->rows, this->cols, [this, matrix](auto i, auto j) {
            return error_threshold(this->Get(i, j) + matrix->Get(i, j));
        });
    }

    void operator+=(const Matrix<T> *matrix) {
        if (!this->matchSize(matrix)) {
            throw DimensionalException();
        }

        this->Map([matrix](auto i, auto j, auto value) {
            return matrix->error_threshold(value + matrix->Get(i, j));
        });
    }

    Matrix<T> *operator-(const Matrix<T> *matrix) const {
        if (!this->matchSize(matrix)) {
            throw DimensionalException();
        }

        return new Matrix<T>(this->rows, this->cols, [this, matrix](auto i, auto j) {
            return error_threshold(this->Get(i, j) - matrix->Get(i, j));
        });
    }

    void operator-=(const Matrix<T> *matrix) {
        if (!this->matchSize(matrix)) {
            throw DimensionalException();
        }

        this->Map([matrix](auto i, auto j, auto value) {
            return matrix->error_threshold(value - matrix->Get(i, j));
        });
    }

    Matrix<T> *operator-() const {
        return new Matrix<T>(this->rows, this->cols, [this](auto i, auto j) { return -this->Get(i, j); });
    }

    Matrix<T> *operator*(Matrix<T> *matrix) const {
        if (this->cols != matrix->rows) {
            throw DimensionalException();
        }
        // NxM * MxK = NxK
        return new Matrix<T>(this->rows, matrix->cols, [this, &matrix](auto i, auto k) {
            T dot_product = 0;
            for (int j = 0; j < this->cols; ++j) {
                dot_product += this->Get(i, j) * matrix->Get(j, k);
            }
            return error_threshold(dot_product);
        });
    }

    Matrix<T> *operator*(const T scalar) const {
        return new Matrix<T>(this->rows, this->cols, [this, scalar](auto i, auto j) {
            return error_threshold(this->Get(i, j) * scalar);
        });
    }

    bool operator==(const Matrix<T> *matrix) const {
        if (!this->matchSize(matrix)) {
            return false;
        }

        for (int i = 0; i < this->rows; ++i) {
            for (int j = 0; j < this->cols; ++j) {
                if (error_threshold(this->Get(i, j) - matrix->Get(i, j)) != 0.0) {
                    return false;
                }
            }
        }

        return true;
    }

    [[nodiscard]] Matrix<T> *Clone() const {
        return new Matrix<T>(this->rows, this->cols, [this](auto i, auto j) { return this->Get(i, j); });
    }

    virtual std::istream &read_from(std::istream &input) {
        input >> this->rows >> this->cols;
        delete this->raw_data;
        this->raw_data = new T[this->cols * this->rows];
        for (int i = 0; i < this->rows; ++i) {
            for (int j = 0; j < this->cols; ++j) {
                input >> this->raw_data[i * this->cols + j];
            }
        }
        return input;
    }

    friend std::istream &operator>>(std::istream &input, Matrix &matrix) {
        return matrix.read_from(input);
    }

    friend std::ostream &operator<<(std::ostream &os, const Matrix *matrix) {
        for (int i = 0; i < matrix->rows; ++i) {
            for (int j = 0; j < matrix->cols - 1; ++j) {
                os << matrix->error_threshold(matrix->Get(i, j))
                   << " ";
            }
            os << matrix->error_threshold(matrix->Get(i, matrix->cols - 1))
               << std::endl;
        }
        return os;
    }

    [[nodiscard]] Matrix<T> *Transpose() const {
        return new Matrix<T>(this->cols, this->rows, [this](auto i, auto j) { return this->Get(j, i); });
    }

    [[nodiscard]] T Determinant() const {
        if (this->cols != this->rows) {
            throw DimensionalException();
        }

        T det = 1;

        Matrix<T> *I = new IdentityMatrix<T>(this->cols);

        auto matrix = this->Clone();
        for (int j = 0; j < matrix->cols; ++j) {
            auto permutation = matrix->PivotByMaximum(j);
            if (*permutation != I) {
                det *= -1;
            }
            delete permutation;

            for (int i = j + 1; i < matrix->rows; ++i) {
                delete matrix->Eliminate(i, j);
            }
        }

        for (int j = 0; j < matrix->cols; ++j) {
            det *= matrix->Get(j, j);
        }

        free({I, matrix});
        return error_threshold(det);
    }

    Matrix<T> *Eliminate(size_t row, size_t col) {
        if (abs(this->Get(row, col)) <= ERROR_THRESHOLD || abs(this->Get(col, col)) <= ERROR_THRESHOLD) {
            return new IdentityMatrix<T>(this->cols);
        }

        T k = -(this->Get(row, col) / this->Get(col, col));

        this->Map([row, col, k, this](auto r, auto c, auto v) {
            if (r == row) {
                return error_threshold(v + k * this->Get(col, c));
            }
            return v;
        });

        return new EliminationMatrix<T>(this->cols, row, col, k);
    }

    Matrix<T> *Inverse() const {
        if (this->cols != this->rows) {
            throw DimensionalException();
        }

        if (this->Determinant() == 0.0) {
            throw SingularMatrixException();
        }

        Matrix<T> *M = this->Clone();
        Matrix<T> *I = new IdentityMatrix<T>(this->cols);

        Matrix<T> *augmented = new Matrix<T>(this->rows, this->cols * 2, [this, I, M](auto i, auto j) {
            if (j >= this->cols) {
                return I->Get(i, j - this->cols);
            }
            return M->Get(i, j);
        });


        for (size_t col = 0; col < this->cols; ++col) {
            auto permutation_matrix = M->PivotByMaximum(col);
            augmented = *permutation_matrix * augmented;
            delete permutation_matrix;

            for (size_t row = col + 1; row < this->rows; ++row) {
                auto elimination_matrix = M->Eliminate(row, col);
                augmented = *elimination_matrix * augmented;
                delete elimination_matrix;
            }
        }

        for (long long col = this->cols - 1; col > -1; --col) {
            for (long long row = col - 1; row > -1; --row) {
                auto elimination_matrix = M->Eliminate(row, col);
                augmented = *elimination_matrix * augmented;
                delete elimination_matrix;
            }
        }

        auto Di = new IdentityMatrix<T>(this->cols);
        for (size_t col = 0; col < this->cols; ++col) {
            if (abs(augmented->Get(col, col)) > Matrix<double>::ERROR_THRESHOLD) {
                Di->Put(col, col, 1 / augmented->Get(col, col));
            }
        }
        augmented = *Di * augmented;

        Matrix<T> *inverse = new Matrix<T>(this->cols, this->rows, [this, augmented](auto i, auto j) {
            return augmented->Get(i, j + this->cols);
        });

        free({augmented, Di, M, I});
        return inverse;
    }

    void SwapRows(size_t row1, size_t row2) {
        auto row1_data = new T[this->cols];
        auto row2_data = new T[this->cols];

        for (int i = 0; i < this->cols; ++i) {
            row1_data[i] = this->Get(row1, i);
            row2_data[i] = this->Get(row2, i);
        }

        this->Map([&row1_data, &row2_data, row1, row2](auto row, auto col, auto old) {
            if (row == row1) {
                return row2_data[col];
            }

            if (row == row2) {
                return row1_data[col];
            }

            return old;
        });

        delete[] row1_data;
        delete[] row2_data;
    }

    void SwapCols(size_t col1, size_t col2) {
        auto col1_data = new T[this->rows];
        auto col2_data = new T[this->rows];

        for (int i = 0; i < this->rows; ++i) {
            col1_data[i] = this->Get(i, col1);
            col2_data[i] = this->Get(i, col2);
        }

        this->Map([&col1_data, &col2_data, col1, col2](auto row, auto col, auto old) {
            if (col == col1) {
                return col2_data[row];
            }

            if (col == col2) {
                return col1_data[row];
            }

            return old;
        });

        delete[] col1_data;
        delete[] col2_data;
    }

    Matrix<T> *PivotByMaximum(size_t column) {
        size_t max_index = column;
        for (size_t i = column + 1; i < this->rows; ++i) {
            if (std::abs(this->Get(i, column)) > std::abs(this->Get(max_index, column))) {
                max_index = i;
            }
        }

        if (column != max_index) {
            this->SwapRows(column, max_index);
            return new PermutationMatrix<T>(this->cols, column, max_index);
        }

        return new IdentityMatrix<T>(this->cols);
    }

    ColumnVector<T>* GetColumn(size_t column) const {
        return new ColumnVector<T>(this->rows, [this, column](auto i) {
            return this->Get(i, column);
        });
    }
};

template<required_accuracy T>
class SquareMatrix : public Matrix<T> {
public:
    explicit SquareMatrix(size_t dim) : Matrix<T>(dim, dim) {}

    SquareMatrix(size_t dim, std::function<T(size_t, size_t)> func) : Matrix<T>(dim, dim, func) {}

    std::istream &read_from(std::istream &input) override {
        input >> this->rows;
        this->cols = this->rows;
        delete this->raw_data;
        this->raw_data = new T[this->cols * this->rows];
        for (int i = 0; i < this->rows; ++i) {
            for (int j = 0; j < this->cols; ++j) {
                input >> this->raw_data[i * this->cols + j];
            }
        }
        return input;
    }
};

template<required_accuracy T>
class IdentityMatrix : public SquareMatrix<T> {
public:
    explicit IdentityMatrix(size_t dim) : SquareMatrix<T>(dim, [](auto i, auto j) { return (T) i == j; }) {}
};

template<required_accuracy T>
class EliminationMatrix : public IdentityMatrix<T> {
public:
    EliminationMatrix(size_t dim, size_t row1, size_t row2, T k) : IdentityMatrix<T>(dim) {
        this->Put(row1, row2, k);
    }

    EliminationMatrix(size_t dim, size_t row1, size_t row2, Matrix<T> *A) : IdentityMatrix<T>(dim) {
        this->Put(row1, row2, -(T) ((double) A->Get(1, 0) / (double) A->Get(0, 0)));
    }
};

template<required_accuracy T>
class PermutationMatrix : public IdentityMatrix<T> {
public:
    PermutationMatrix(size_t dim, size_t row1, size_t row2) : IdentityMatrix<T>(dim) {
        this->SwapRows(row1, row2);
    }
};

template<required_accuracy T>
class ColumnVector : public Matrix<T> {
public:
    explicit ColumnVector(size_t dim) : Matrix<T>(dim, 1) {}

    ColumnVector(size_t dim, std::function<T(size_t)> func) : Matrix<T>(dim, 1, [func](auto i, auto j) {
        return func(i);
    }) {}

    std::istream &read_from(std::istream &input) override {
        input >> this->rows;
        this->cols = 1;
        delete this->raw_data;
        this->raw_data = new T[this->rows];
        for (int i = 0; i < this->rows; ++i) {
            input >> this->raw_data[i];
        }
        return input;
    }

    T norm() {
        T x = 0;
        for (int i = 0; i < this->rows; ++i) {
            x += std::pow(this->raw_data[i], 2);
        }
        return this->error_threshold(std::sqrt(x));
    }
};


template<required_accuracy T>
class Calculator {
public:
    static ColumnVector<T> *JacobiMethod(Matrix<T> *A, ColumnVector<T> *b, T approximation_accuracy) {
        bool diagonal_dominance = true;
        for (int i = 0; i < A->Rows(); ++i) {
            T sum = 0;
            for (int j = 0; j < A->Cols(); ++j) {
                if (i != j) {
                    sum += std::abs(A->Get(i, j));
                }
            }
            diagonal_dominance &= A->Get(i, i) > sum;
        }


        if (!diagonal_dominance) {
            std::cout << "The method is not applicable!\n";
            return new ColumnVector<T>(b->Rows(), [](auto x) {
                return (T) 0;
            });;
        }


        Matrix<T> *L = new SquareMatrix<T>(A->Rows(), [&A](auto i, auto j) {
            if (j < i) {
                return A->Get(i, j);
            }
            return (T) 0;
        });
        Matrix<T> *U = new SquareMatrix<T>(A->Rows(), [&A](auto i, auto j) {
            if (j > i) {
                return A->Get(i, j);
            }
            return (T) 0;
        });
        Matrix<T> *Di = new SquareMatrix<T>(A->Rows(), [&A](auto i, auto j) {
            if (j == i) {
                return 1 / A->Get(i, j);
            }
            return (T) 0;
        });
        Matrix<T> *LU = *L + U;
        Matrix<T> *nDi = -*Di;
        Matrix<T> *alpha = *nDi * LU;
        Matrix<T> *beta = *Di * b;

        std::cout << "alpha:" << std::endl << alpha;
        std::cout << "beta:" << std::endl << beta;

        auto *x = (ColumnVector<T>*) beta->Clone();

        int k = 0;

        T accuracy;
        do {
            // print x_k
            std::cout << "x(" << k << "):" << std::endl << x;

            // x_{k + 1}
            Matrix<T> *temp0 = x;
            Matrix<T> *temp1 = *alpha * x;
            Matrix<T> *temp2 = *temp1 + beta;
            x = (ColumnVector<T> *) temp2->Clone();
            // e = |x_k+1 - x_k|
            auto *temp4 = (ColumnVector<T>*) (*x - temp0);
            accuracy = temp4->norm();
            std::cout << "e: " << accuracy << std::endl;

            k++;
            free({temp0, temp1, temp2, temp4});
        } while (accuracy > approximation_accuracy);

        std::cout << "x(" << k << "):" << std::endl << x;

        free({L, U, Di, nDi, alpha, beta});
        return (ColumnVector<T> *) x;
    }

    static ColumnVector<T> *SeidelMethod(Matrix<T> *A, ColumnVector<T> *b, T approximation_accuracy) {
        bool diagonal_dominance = true;
        for (int i = 0; i < A->Rows(); ++i) {
            T sum = 0;
            for (int j = 0; j < A->Cols(); ++j) {
                if (i != j) {
                    sum += std::abs(A->Get(i, j));
                }
            }
            diagonal_dominance &= A->Get(i, i) > sum;
        }


        if (!diagonal_dominance) {
            std::cout << "The method is not applicable!\n";
            return new ColumnVector<T>(b->Rows(), [](auto x) {
                return (T) 0;
            });;
        }

        Matrix<T> *L = new SquareMatrix<T>(A->Rows(), [&A](auto i, auto j) {
            if (j < i) {
                return A->Get(i, j);
            }
            return (T) 0;
        });
        Matrix<T> *U = new SquareMatrix<T>(A->Rows(), [&A](auto i, auto j) {
            if (j > i) {
                return A->Get(i, j);
            }
            return (T) 0;
        });
        Matrix<T> *Di = new SquareMatrix<T>(A->Rows(), [&A](auto i, auto j) {
            if (j == i) {
                return 1 / A->Get(i, j);
            }
            return (T) 0;
        });

        Matrix<T> *LU = *L + U;

        Matrix<T> *nDi = -*Di;

        Matrix<T> *alpha = *nDi * LU;
        Matrix<T> *beta = *Di * b;

        Matrix<T>* C = new SquareMatrix<T>(alpha->Rows(), [&alpha](auto i, auto j) {
            if (j > i) {
                return alpha->Get(i, j);
            }
            return (T) 0;
        });
        Matrix<T>* B = new SquareMatrix<T>(alpha->Rows(), [&alpha](auto i, auto j) {
            if (j <= i) {
                return alpha->Get(i, j);
            }
            return (T) 0;
        });

        Matrix<T>* I = new IdentityMatrix<T>(alpha->Rows());
        Matrix<T>* IB = *I - B;
        Matrix<T>* IBi = IB->Inverse();
        Matrix<T>* IBiC = *IBi * C;

        Matrix<T>* IBib = *IBi * beta;

        Matrix<T>* IBiCt = IBiC->Transpose();

        std::cout << "beta:" << std::endl << beta;
        std::cout << "alpha:" << std::endl << alpha;
        std::cout << "B:" << std::endl << B;
        std::cout << "C:" << std::endl << C;
        std::cout << "I-B:" << std::endl << IB;
        std::cout << "(I-B)_-1:" << std::endl << IBi;

        // auto* x = (ColumnVector<T>*) IBib->Clone();
        auto* x = (ColumnVector<T>*) beta->Clone();

        T accuracy;
        std::cout << "x(" << 0 << "):" << std::endl << x;


        int k = 1;
        do {
            Matrix<T>* temp0 = x->Clone();

            // Version from lecture
            // for (int i = 0; i < x->Rows(); ++i) {
            //    Matrix<T>* temp1 = IBiCt->GetColumn(i);
            //    Matrix<T>* temp2 = temp1->Transpose();
            //    Matrix<T>* temp3 = *temp2*x;
            //    x->Put(i, 0, temp3->Get(0, 0) + IBib->Get(i, 0));
            //    free({temp1, temp2, temp3});
            // }

            Matrix<T>* temp1 = *C*temp0;
            Matrix<T>* temp2 = *temp1 + beta;
            x = (ColumnVector<T>*) (*IBi * temp2);

            // e = |x_k+1 - x_k|
            auto *temp4 = (ColumnVector<T>*) (*x - temp0);
            accuracy = temp4->norm();
            std::cout << "e: " << accuracy << std::endl;

            std::cout << "x(" << k << "):" << std::endl << x;

            k++;
            free({temp0, temp4});
        } while (accuracy > approximation_accuracy);

        free({L, U, Di, LU, nDi, alpha, beta, C, B, I, IB, IBi, IBiC, IBib, IBiCt});

        return x;
    }
};

int main() {
    // Specify Output Format
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(4);

    // Read Samples
    size_t m;
    std::cin >> m;
    Matrix<double> *ts = new ColumnVector<double>(m);
    Matrix<double> *b = new ColumnVector<double>(m);
    for (int i = 0; i < m; ++i) {
        double ti, bi;
        std::cin >> ti >> bi;
        ts->Put(i, 0, ti);
        b->Put(i, 0, bi);
    }
    // Read Polynomial Degree
    size_t n;
    std::cin >> n;

    // Generate Matrix A
    auto *A = new Matrix<double>(m, n+1, [ts](auto i, auto j) {
        return pow(ts->Get(i, 0), j);
    });

    // Find Model
    auto At = A->Transpose();
    auto AtA = *At * A;
    auto AtAi = AtA->Inverse();
    auto Atb = *At * b;
    auto x = *AtAi * Atb;

    // Report Steps
    std::cout << "A:\n" << A;
    std::cout << "A_T*A:\n" << AtA;
    std::cout << "(A_T*A)^-1:\n" << AtAi;
    std::cout << "A_T*b:\n" << Atb;
    std::cout << "x~:\n" << x;

    // Free Memory
    free({ts, A, b, At, AtA, AtAi, Atb, x});
    return 0;
}