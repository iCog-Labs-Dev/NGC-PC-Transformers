from jax import numpy as jnp
from ngclearn import Compartment, compilable
from ngclearn.components.jaxComponent import JaxComponent


class ResidualComponent(JaxComponent):
    """
    A residual / skip-connection component for Predictive Coding Networks (PCNs).

    Forward pass:
        outputs = x1 + x2

    Reverse error pass:
        dx1 = dmu
        dx2 = dmu
    """

    def __init__(self, name, shape, **kwargs):
        super().__init__(name, **kwargs)
        self.shape = shape

        # Forward compartments
        self.x1 = Compartment(jnp.zeros(shape))
        self.x2 = Compartment(jnp.zeros(shape))
        self.outputs = Compartment(jnp.zeros(shape))

        # Reverse error compartments
        self.dmu = Compartment(jnp.zeros(shape))
        self.dx1 = Compartment(jnp.zeros(shape))
        self.dx2 = Compartment(jnp.zeros(shape))

    @compilable
    def advance_state(self):
        x1 = self.x1.get()
        x2 = self.x2.get()
        self.outputs.set(x1 + x2)

        dmu = self.dmu.get()
        self.dx1.set(dmu)
        self.dx2.set(dmu)

    @compilable
    def reset(self):
        zeros = jnp.zeros(self.shape)
        self.x1.set(zeros)
        self.x2.set(zeros)
        self.outputs.set(zeros)
        self.dmu.set(zeros)
        self.dx1.set(zeros)
        self.dx2.set(zeros)
