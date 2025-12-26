from jax import  numpy as jnp
from ngclearn.components.jaxComponent import JaxComponent
from ngclearn import Compartment
from ngclearn import compilable

# from ngcsimlib.compilers.process import transition



class ReshapeComponent(JaxComponent):
    """Component that reshapes tensors for ngc-learn wiring"""
    
    def __init__(self, name, input_shape, output_shape, **kwargs):
        super().__init__(name, **kwargs)
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.inputs = Compartment(jnp.zeros(input_shape))
        self.outputs = Compartment(jnp.zeros(output_shape))
    
    @compilable
    def advance_state(self):
        inputs=self.inputs.get()
        output_shape=self.output_shape.get()
        output=inputs.reshape(output_shape)
        self.outputs.set(output)
    
    
    @compilable
    def reset(self):
        input_shape=self.input_shape.get()
        output_shape=self.output_shape.get()
        input=jnp.zeros(input_shape)
        output=jnp.zeros(output_shape)
        self.inputs.set(input)
        self.outputs.set(output)