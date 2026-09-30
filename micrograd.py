import math


class Value:
    def __init__(self, val, requires_grad=True, grad_fn=None, grad=0.0, is_leaf=True, child1=None, child2=None):
        self.value = val
        self.requires_grad = requires_grad
        self.grad_fn = grad_fn
        self.grad = grad
        self.is_leaf = is_leaf

        self.child1 = child1
        self.child2 = child2

    def __str__(self):
        return f"{self.value}"

    def __add__(self, other):
        result = self.value + other.value
        new_obj = Value(val=result, is_leaf=False, child1=self, child2=other)
        new_obj.grad_fn = new_obj.add_backwards
        return new_obj

    def __mul__(self, other):
        result = self.value * other.value
        new_obj = Value(val=result, is_leaf=False, child1=self, child2=other)
        new_obj.grad_fn = new_obj.mult_backwards
        return new_obj

    def __truediv__(self, other):
        result = self.value / (other.value + 1e-8) # stop div by 0
        new_obj = Value(val=result, is_leaf=False, child1=self, child2=other)
        new_obj.grad_fn = new_obj.div_backwards
        return new_obj

    def __pow__(self, other):
        result = self.value ** other.value
        new_obj = Value(val=result, is_leaf=False, child1=self, child2=other)
        new_obj.grad_fn = new_obj.pow_backwards
        return new_obj

    def __sub__(self, other):
        result = self.value - other.value
        new_obj = Value(val=result, is_leaf=False, child1=self, child2=other)
        new_obj.grad_fn = new_obj.sub_backwards
        return new_obj

    # backwards functions return child1, child2
    def add_backwards(self, parent_grad):
        return parent_grad, parent_grad

    def mult_backwards(self, parent_grad):
        return parent_grad * self.child2.value, parent_grad * self.child1.value

    def div_backwards(self, parent_grad):
        c2 = -(self.child1.value)/(self.child2.value)**2
        return parent_grad/self.child2.value, parent_grad * c2

    def pow_backwards(self, parent_grad):
        c1_grad = self.child2.value * (self.child1.value ** (self.child2.value - 1))
        c2_grad = self.value * math.log(self.child1.value)
        return parent_grad * c1_grad, parent_grad * c2_grad

    def sub_backwards(self, parent_grad):
        return parent_grad, -parent_grad


    def get_ordered_nodes(self):
        # build topological order
        visited = set()
        topo = []

        def topo_dfs(v):
            if v not in visited:
                visited.add(v)
                if v.child1:
                    topo_dfs(v.child1)
                if v.child2:
                    topo_dfs(v.child2)
                topo.append(v)

        topo_dfs(self)
        return reversed(topo)

    
    def backward(self, parent_grad=None):
        
        ordered_nodes = self.get_ordered_nodes()

        
        if parent_grad is None:
            self.grad = 1.0
        else:
            self.grad = parent_grad

        # propagate back
        for node in ordered_nodes:
            if node.is_leaf or node.grad_fn is None:
                continue

            c1_grad, c2_grad = node.grad_fn(parent_grad=node.grad)

            if node.child1:
                node.child1.grad += c1_grad
            if node.child2:
                node.child2.grad += c2_grad


a = Value(2)
b = Value(3)

c = Value(4)
d = Value(5)

e = a * b

f = c + d

g = e/f

g.backward()