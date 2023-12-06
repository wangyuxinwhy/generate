import itertools
from functools import reduce
from typing import Callable, Iterable, Iterator, List, Optional, TypeVar

T = TypeVar('T')
U = TypeVar('U')


class IterChain(Iterator[T]):
    """
    IterChain is a utility class for performing chainable operations on iterables in Python.

    This class wraps an iterable object and provides a range of methods for common
    iterable manipulations, such as map, filter, and reduce, in a fluent and chainable manner.
    It is inspired by the capabilities of Rust's iterators and other functional programming paradigms.

    Each method in IterChain returns a new IterChain instance, allowing for the methods
    to be chained together to perform complex transformations in a readable and expressive way.

    Methods:
        - map(func): Applies a function to every item of the iterable.
        - filter(func): Yields items of the iterable for which the function returns true.
        - reduce(func, initial): Reduces the iterable to a single value using the provided function.
        - takewhile(predicate): Yields items as long as the predicate is true.
        - skip(n): Skips the first n items of the iterable.
        - take(n): Takes the first n items of the iterable.
        - flatten(): Flattens a nested iterable into a single iterable.
        - chunk(size): Breaks the iterable into chunks of the given size.

    Example Usage:
        # Create an IterChain instance
        numbers = IterChain([1, 2, 3, 4, 5])

        # Chain multiple operations
        result = numbers.filter(lambda x: x % 2 == 0).map(lambda x: x * x).take(3)

        # Iterate over the result
        for num in result:
            print(num)

    Note:
        IterChain instances are iterators and can be iterated only once.
        To reuse the same data, create a new IterChain instance.
    """

    def __init__(self, iterable: Iterable[T]) -> None:
        self._iterator = iter(iterable)

    def __iter__(self) -> Iterator[T]:
        return self

    def __next__(self) -> T:
        return next(self._iterator)

    def map(self, func: Callable[[T], U]) -> 'IterChain[U]':
        return IterChain(map(func, self))

    def filter(self, func: Callable[[T], bool]) -> 'IterChain[T]':
        return IterChain(filter(func, self))

    def reduce(self, func: Callable[[U, T], U], initial: Optional[U] = None) -> U:
        if initial is not None:
            return reduce(func, self, initial)
        return reduce(func, self)  # type: ignore

    def takewhile(self, predicate: Callable[[T], bool]) -> 'IterChain[T]':
        return IterChain(itertools.takewhile(predicate, self))

    def skip(self, n: int) -> 'IterChain[T]':
        return IterChain(itertools.islice(self, n, None))

    def take(self, n: int) -> 'IterChain[T]':
        return IterChain(itertools.islice(self, n))

    def flatten(self) -> 'IterChain':
        return IterChain(itertools.chain.from_iterable(item if isinstance(item, Iterable) else [item] for item in self))

    def chunk(self, size: int) -> 'IterChain[List[T]]':
        return IterChain(iter(lambda: list(itertools.islice(self, size)), []))
