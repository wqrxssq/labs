from sympy import Matrix, randMatrix

from hashlib import md5


from Algebra.src.solution import AbelianGroup, PersonalInfo, smith_normal_form


def test_personal_info():
    PI = (
        PersonalInfo["группа"] + PersonalInfo["Фамилия"] + PersonalInfo["Имя"]
    ).encode("utf-8")
    hash = md5(PI).hexdigest()
    print(hash)
    found = False
    with open("students_hashes.txt") as hashes:
        for line in hashes.readlines():
            if hash == line.strip():
                found = True
                break
    assert found, f"Студент не найден в списке, {PersonalInfo}, хэш {hash}"


def test_smith_normal_form():
    def check_snf(A, invariant_factors=None):
        S, D, T = smith_normal_form(A)
        assert D is not A, (
            "smith_normal_form не должна менять матрицу-аргумент или возвращать ее"
        )
        assert all(a.is_integer for a in S) and all(a.is_integer for a in T)
        u = D.diagonal() if 0 not in D.shape else []
        for i in range(len(u) - 1):
            if u[i] != 0:
                assert u[i + 1] % u[i] == 0
        assert S * A * T == D
        if invariant_factors is not None:
            assert list(u) == invariant_factors

    check_snf(Matrix([[2, 3], [5, 7]]), [1, 1])
    check_snf(Matrix([[2, 3], [5, 7], [1, 1]]), [1, 1])
    check_snf(Matrix([[3, 5, -4], [9, 7, 0]]), [1, 4])
    check_snf(Matrix([[1, 3], [-2, 1]]), [1, 7])
    for i in range(200):
        check_snf(randMatrix(3, 3, -20, 20))


def test_invariant_factors():
    A = AbelianGroup(
        3,
        Matrix([
            [1, 3],
            [5, 8],
            [0, 0],
        ]),
    )
    assert A.invariant_factors() == [1, 7, 0]
    A = AbelianGroup(
        2,
        Matrix([
            [1, 3, 0],
            [5, 8, 0],
        ]),
    )
    assert A.invariant_factors() == [1, 7]


def test_is_trivial():
    A = AbelianGroup(
        3,
        Matrix([
            [1, 3],
            [5, 8],
            [0, 0],
        ]),
    )
    assert not A.is_trivial()
    A = AbelianGroup(
        2,
        Matrix([
            [1, 0, 3],
            [0, 2, 1],
        ]),
    )
    assert A.is_trivial()
    A = AbelianGroup(
        2,
        Matrix([
            [1, 3],
            [5, 8],
        ]),
    )
    assert not A.is_trivial()


def test_is_cyclic():
    A = AbelianGroup(3, Matrix([[2, 0, 0], [0, 2, 0], [0, 0, 2]]))
    assert not A.is_cyclic()
    B = AbelianGroup(2, Matrix([[1], [0]]))
    assert B.is_cyclic()
    C = AbelianGroup(1, Matrix([[4]]))
    assert C.is_cyclic()


def test_is_free():
    for n in range(1, 5):
        assert AbelianGroup(n).is_free()
    assert AbelianGroup(3, Matrix([[1], [1], [1]])).is_free()
    assert not AbelianGroup(2, Matrix([[2, 1], [0, 1]])).is_free()
    assert AbelianGroup(2, Matrix.eye(2)).is_free()  # trivial is free


def test_rank():
    for n in range(1, 5):
        assert AbelianGroup(n).rank() == n
    assert AbelianGroup(3, Matrix([[1], [0], [0]])).rank() == 2
    assert AbelianGroup(2, Matrix([[2, 1], [0, 1]])).rank() is None


def test_is_finite():
    assert AbelianGroup(0).is_finite()
    assert not AbelianGroup(2).is_finite()
    assert AbelianGroup(2, Matrix([[1, 0], [1, 2]])).is_finite()


def test_size():
    A = AbelianGroup(3, Matrix([[5, 0, 5], [7, 2, 0], [0, 0, 1]]))
    assert A.size() == 10

    B = AbelianGroup(3, Matrix([[4, 0, 0], [0, 5, 0], [0, 0, 2]]))
    assert B.size() == 4 * 5 * 2

    assert AbelianGroup(0).size() == 1
    assert AbelianGroup(1).size() == 0


def test_ord():
    K = Matrix([
        [1, 3],
        [5, 8],
        [0, 0],
    ])
    A = AbelianGroup(3, K)
    assert A.ord(Matrix([1, 0, 0])) == 7
    assert A.ord(Matrix([2, 10, 0])) == 1
    assert A.ord(Matrix([2, 1, 1])) == 0

    K = Matrix([
        [4, 8],
        [2, 10],
        [-100, 120],
    ])
    A = AbelianGroup(3, K)
    assert A.ord(Matrix([32, 31, 0])) == 2
    assert A.ord(Matrix([2, 1, -50])) == 2

    v = Matrix([1, 9])

    A = AbelianGroup(2, Matrix([[308, 2456], [103, 821]]))
    assert A.ord(v) == 100

    A = AbelianGroup(2, Matrix([[8000000001, 10000000001], [36000000004, 45000000004]]))
    assert A.ord(v) == 200000000

    A = AbelianGroup(
        2, Matrix([[48000000000035, 6000000000005], [8000000000007, 1000000000001]])
    )
    assert A.ord(v) == 250000000000


def test_tor():
    K = Matrix([
        [1, 3],
        [5, 8],
        [0, 0],
    ])
    A = AbelianGroup(3, K)
    A_Tor = A.tor()
    assert A_Tor.size() == 7


def test_zero_dim():
    A = AbelianGroup(0)
    assert A.is_free()
    assert A.is_trivial()
    assert A.is_cyclic()
    assert A.is_finite()


if __name__ == "__main__":
    # ручной запуск тестов
    test_personal_info()
    test_smith_normal_form()
    test_invariant_factors()
    test_is_trivial()
    test_is_cyclic()
    test_is_free()
    test_rank()
    test_is_finite()
    test_size()
    test_ord()
    test_tor()
    test_zero_dim()
    print("Все тесты пройдены")
