from functools import cache
from typing import List, Optional, Tuple
from typing_extensions import Self

from sympy import Matrix, eye, zeros, lcm, oo
from sympy.matrices.dense import MutableDenseMatrix
from sympy.matrices.immutable import ImmutableMatrix
from sympy.matrices.matrixbase import MatrixBase
from sympy.core.numbers import Infinity



# впишите свои данные как в ведомости
PersonalInfo = {
    "группа": "",
    "Фамилия": "",
    "Имя": "",
}


def smith_normal_form(A: MatrixBase) -> Tuple[Matrix, Matrix, Matrix]:
    """
    Для ненулевой целочисленной матрицы A размера n*k возвращает тройку S, D, T, где
     S - обратимая над Z целочисленная матрица n*n,
     D - диагональная целочисленная матрица n*k, в которой каждый диагональный элемент делит следующий,
     T - обратимая над Z целочисленная матрица k*k,
    причём D = S*A*T
    """
    raise NotImplementedError


class AbelianGroup:
    """
    Абелева группа G строится как фактор свободной группы Z^n по своей подгруппе ker.
    """

    def __init__(self, n: int, K: MatrixBase | None = None):
        """
        Параметры:
            n:             неотрицательное число, равное рангу группы Z^n
            K:             матрица n*k, где столбцы - соотношения, то есть порождающие подгруппы ker.
        """
        self.n = n
        K = K if K is not None else zeros(self.n, 0)
        self.K = ImmutableMatrix(K)
        assert self.K.shape[0] == self.n

        # делаем self.K непустым, чтобы работал .diagonal()
        if self.K.shape[1] == 0:
            self.K = zeros(self.n, 1)

    @cache
    def _SNF(self) -> Tuple[Matrix, Matrix, Matrix]:
        """
        Возвращает нормальную форму Смита для матрицы `self.K`.
        Если `self.K` пуста, то возвращает разложение с пустой матрицей D соответствующего размера.
        """
        return smith_normal_form(self.K)

    def invariant_factors(self) -> List[int]:
        """
        Возвращает список инвариантных множителей u_1,..,u_m (являются натуральными числами), дополненный нулями до длины `self.n`.
        Нули соответствуют свободным множителям, см. следствие 1 лекции 5.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"AbelianGroup({self.n}, kernel_gens={self.K})"

    def is_trivial(self) -> bool:
        """
        Возвращает `True`, если группа тривиальна (т.е. состоит из нейтрального элемента), и `False` иначе.
        """
        raise NotImplementedError

    def is_cyclic(self) -> bool:
        """
        Возвращает `True`, если группа циклическая, и `False` иначе
        """
        raise NotImplementedError

    def is_free(self) -> bool:
        """
        Возвращает `True`, если группа свободная, и `False` иначе
        """
        raise NotImplementedError

    def rank(self) -> Optional[int]:
        """
        Возвращает ранг группы, если она свободная, и `None` иначе
        """
        raise NotImplementedError

    def is_finite(self) -> bool:
        """
        Возвращает `True`, если группа конечная, и `False` иначе
        """
        raise NotImplementedError

    def size(self) -> int:
        """
        Возвращает размер группы, если она конечна, и 0 иначе
        """
        raise NotImplementedError

    def ord(self, v: Matrix) -> int:
        """
        Возвращает порядок элемента v, если он конечен, и 0 иначе
        """
        raise NotImplementedError

    def tor(self) -> "AbelianGroup":
        """
        Возвращает подгруппу кручения (как абстрактную группу)
        """
        raise NotImplementedError
