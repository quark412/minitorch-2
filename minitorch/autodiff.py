from dataclasses import dataclass
from typing import Any, Iterable, Tuple  # , List

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    i = arg

    lplus = list(vals)
    lplus[i] = lplus[i] + epsilon

    lminus = list(vals)
    lminus[i] = lminus[i] - epsilon

    num = f(*lplus) - f(*lminus)
    denom = 2.0 * epsilon

    return num / denom


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # print("topological sort called on " + str(variable.unique_id))
    visited = {}
    l = []

    def visit(node: Variable) -> None:
        # print("now visiting node " + str(node.unique_id))

        if node.unique_id in visited:
            # print(str(node.unique_id) + " is already visited")
            return
        elif node.is_leaf():
            pass
            # print(str(node.unique_id) + " is a leaf")
        elif node.is_constant():
            pass
        else:
            if node.history is not None:
                parents = node.parents
                # print("The parents of {} are: {}".format(str(node.unique_id), str([x.unique_id for x in parents]))
                for p in parents:
                    visit(p)
            # else:
            # pass
            # print(str(node.unique_id) + " is a problem child.")
        if node.is_constant() is False:
            l.append(node)
        # print(node)
        visited[node.unique_id] = True
        return

    visit(variable)
    l.reverse()  # this reverses in place!

    return l

    """
    def collective_parents(siblings: Iterable[Variable]) -> Iterable[Variable]:
        # gives nonredundant parents of current layer
        # kinda hacky, probably only works if the graph is layered
        d = {}
        for sib in siblings:
            if (sib.is_leaf == False):
                for par in sib.parents:
                    new_id = par.unique_id
                    d[new_id] = par
                # add parents + unique ids to dict
        return list(d.values())

    l = []
    current_layer = [variable]
    while len(current_layer) > 0:
        print("The current layer is as follows")
        print(current_layer)
        for v in current_layer:
            if (v.is_constant == False):
                l.append(v) # add everything in the current layer to l if it isn't constant
        current_layer = collective_parents(current_layer)
    return l
    """


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # print("backprop called on " + str(variable))

    computation_graph = topological_sort(variable)
    d = {variable.unique_id: deriv}
    # print("derivative table is: " + str(d))

    for node in computation_graph:

        if node.is_leaf():
            # print("leaf found")
            # print("d[node] is " + str(d[node.unique_id]))
            node.accumulate_derivative(d[node.unique_id])
        else:
            d_out = d[node.unique_id]
            # print("current node is {} and d_out is {}".format(str(node.unique_id), str(d_out)))
            parents_and_their_derivatives = node.chain_rule(d_out)
            for a, b in parents_and_their_derivatives:
                # print("The parent is {} and the derivative is {}".format(str(a), str(b))
                if a.unique_id in d:
                    d[a.unique_id] = d[a.unique_id] + b
                else:
                    d[a.unique_id] = b
    return


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
