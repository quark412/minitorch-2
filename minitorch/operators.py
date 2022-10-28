"""
Collection of the core mathematical operators used throughout the code base.
"""

import math
from typing import Callable, Iterable

# ## Task 0.1
#
# Implementation of a prelude of elementary functions.


def mul(x: float, y: float) -> float:
    "$f(x, y) = x * y$"
    """
    Multiplication.

    Args:
        x: A float.
        y: A float.

    Returns:
        Product of x and y.
    """
    return x * y
    # raise NotImplementedError("Need to implement for Task 0.1")


def id(x: float) -> float:
    "$f(x) = x$"
    """
    Identity.

    Args:
        x: A float.

    Returns:
        x.
    """
    return x
    # raise NotImplementedError("Need to implement for Task 0.1")


def add(x: float, y: float) -> float:
    "$f(x, y) = x * y$"
    """
    Addition.

    Args:
        x: A float.
        y: A float.

    Returns:
        Sum of x and y.
    """
    return x + y
    # raise NotImplementedError("Need to implement for Task 0.1")


def neg(x: float) -> float:
    "$f(x) = -x$"
    """
    Negation.

    Args:
        x: A float.

    Returns:
        -x.
    """
    return -1.0 * x
    # raise NotImplementedError("Need to implement for Task 0.1")


def lt(x: float, y: float) -> float:
    "$f(x) =$ 1.0 if x is less than y else 0.0"
    """
    Less than.

    Args:
        x: A float.
        y: A float.

    Returns:
        1.0 if x < y, else 0.0.
    """
    if x < y:
        return 1.0
    else:
        return 0.0
    # raise NotImplementedError("Need to implement for Task 0.1")


def eq(x: float, y: float) -> float:
    "$f(x) =$ 1.0 if x is equal to y else 0.0"
    if x == y:
        return 1.0
    else:
        return 0.0
    # raise NotImplementedError("Need to implement for Task 0.1")


def max(x: float, y: float) -> float:
    "$f(x) =$ x if x is greater than y else y"
    if x > y:
        return x
    else:
        return y
    # raise NotImplementedError("Need to implement for Task 0.1")


def leq(x: float, y: float) -> float:
    "$f(x) =$ 1.0 if x is less than or equal to y else 0.0"
    return max(lt(x, y), eq(x, y))


def is_close(x: float, y: float) -> float:
    "$f(x) = |x - y| < 1e-2$"
    if abs(x - y) < 1e-2:
        return 1.0
    else:
        return 0.0
    # raise NotImplementedError("Need to implement for Task 0.1")


def sigmoid(x: float) -> float:
    r"""
    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$

    (See https://en.wikipedia.org/wiki/Sigmoid_function )

    Calculate as

    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$

    for stability.
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))
    # raise NotImplementedError("Need to implement for Task 0.1")


def relu(x: float) -> float:
    """
    $f(x) =$ x if x is greater than 0, else 0

    (See https://en.wikipedia.org/wiki/Rectifier_(neural_networks) .)
    """
    if x >= 0:
        return x
    else:
        return 0.0
    # raise NotImplementedError("Need to implement for Task 0.1")


EPS = 1e-6


def log(x: float) -> float:
    "$f(x) = log(x)$"
    return math.log(x + EPS)


def exp(x: float) -> float:
    "$f(x) = e^{x}$"
    return math.exp(x)


def inv(x: float) -> float:
    "$f(x) = 1/x$"
    return 1.0 / x
    # raise NotImplementedError("Need to implement for Task 0.1")


def inv_back(x: float, d: float) -> float:
    "If $f(x) = 1/x$ compute $d \times f'(x)$"
    return d * neg(inv(x**2))
    # raise NotImplementedError("Need to implement for Task 0.1")


def log_back(x: float, d: float) -> float:
    "If $f = log$ as above, compute $d \times f'(x)$"
    return d * inv(x)
    # raise NotImplementedError("Need to implement for Task 0.1")


def relu_back(x: float, d: float) -> float:
    "If $f = relu$ compute $d \times f'(x)$"
    if x >= 0:
        return d
    else:
        return 0.0
    # NOTE might be wrong if x = 0, not sure
    # raise NotImplementedError("Need to implement for Task 0.1")


# ## Task 0.3

# Small practice library of elementary higher-order functions.


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """
    Higher-order map.

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: Function from one value to one value.

    Returns:
        A function that takes a list, applies `fn` to each element, and returns a
         new list
    """

    def f(l: Iterable[float]) -> Iterable[float]:
        new_l = list()
        for i in l:
            new_l.append(fn(i))
        return new_l  # outputs list, wanted to preserve the type but it doesn't work segun mypy

    return f


def negList(ls: Iterable[float]) -> Iterable[float]:
    "Use `map` and `neg` to negate each element in `ls`"
    return map(neg)(ls)


def zipWith(
    fn: Callable[[float, float], float]
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """
    Higher-order zipwith (or map2).

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: combine two values

    Returns:
        Function that takes two equally sized lists `ls1` and `ls2`, produce a new list by
         applying fn(x, y) on each pair of elements.

    """

    def f(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        new_l = list()
        for i, j in zip(ls1, ls2):
            new_l.append(fn(i, j))
        return new_l

    return f


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    "Add the elements of `ls1` and `ls2` using `zipWith` and `add`"
    return zipWith(add)(ls1, ls2)


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    r"""
    Higher-order reduce.

    Args:
        fn: combine two values
        start: start value $x_0$

    Returns:
        Function that takes a list `ls` of elements
         $x_1 \ldots x_n$ and computes the reduction :math:`fn(x_3, fn(x_2,
         fn(x_1, x_0)))`
    """

    def f(l: Iterable[float]) -> float:
        y = start
        for x in l:
            y = fn(x, y)
        return y

    return f


def sum(ls: Iterable[float]) -> float:
    "Sum up a list using `reduce` and `add`."
    return reduce(add, 0.0)(ls)


def prod(ls: Iterable[float]) -> float:
    "Product of a list using `reduce` and `mul`."
    # TODO: Implement for Task 0.3.
    return reduce(mul, 1.0)(ls)
