from jax import numpy as jnp
from ngclearn import Compartment, compilable
from ngclearn.components.jaxComponent import JaxComponent


class ResidualComponent(JaxComponent):
    """
    A residual / skip-connection component for Predictive Coding Networks (PCNs).

    Forward pass:
        outputs = x1 + x2

    Reverse error pass:
        dx1 receives bottom-up pre_out current
        dx2 receives top-down dmu error
        dx = dx1 + dx2 (summed derivative passed to rate cell j)
    """

    def __init__(self, name, shape, **kwargs):
        super().__init__(name, **kwargs)
        self.shape = shape

        # Forward compartments
        self.x1 = Compartment(jnp.zeros(shape))
        self.x2 = Compartment(jnp.zeros(shape))
        self.outputs = Compartment(jnp.zeros(shape))

        # Reverse error / derivative compartments
        self.dx1 = Compartment(jnp.zeros(shape))
        self.dx2 = Compartment(jnp.zeros(shape))
        self.dx = Compartment(jnp.zeros(shape))

    @compilable
    def advance_state(self):
        x1 = self.x1.get()
        x2 = self.x2.get()
        self.outputs.set(x1 + x2)

        dx1 = self.dx1.get()
        dx2 = self.dx2.get()
        self.dx.set(dx1 + dx2)

    @compilable
    def reset(self):
        zeros = jnp.zeros(self.shape)
        self.x1.set(zeros)
        self.x2.set(zeros)
        self.outputs.set(zeros)
        self.dx1.set(zeros)
        self.dx2.set(zeros)
        self.dx.set(zeros)
