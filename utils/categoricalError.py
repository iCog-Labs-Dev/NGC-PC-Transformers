# %%
from ngclearn.components.jaxComponent import JaxComponent
from jax import numpy as jnp, jit
from ngclearn import compilable 
from ngclearn import Compartment 

class CategoricalErrorCell(JaxComponent): 
    """
    A Categorical error cell - this computes the KL-Divergence between 
    a target distribution and a predicted distribution (mu).

    | --- Cell Input Compartments: ---
    | mu - predicted probability distribution (usually from a Softmax)
    | target - desired/goal distribution (e.g., one-hot vector)
    | modulator - modulation signal (scaling)
    | mask - binary/gating mask 
    | --- Cell Output Compartments: ---
    | L - KL-Divergence: D_KL(target || mu)
    | dmu - derivative of L w.r.t. mu (the error signal)
    | dtarget - derivative of L w.r.t. target
    """
    def __init__(self, name, n_units, batch_size=1, shape=None, eps=1e-8, **kwargs):
        super().__init__(name, **kwargs)

        ## Layer Size Setup
        _shape = (batch_size, n_units)
        if shape is None:
            shape = (n_units,)
        else:
            _shape = (batch_size, shape[0], shape[1], shape[2])
            
        self.shape = shape
        self.n_units = n_units
        self.batch_size = batch_size
        self.eps = eps # Stability constant to prevent log(0)

        ## Compartment setup
        restVals = jnp.zeros(_shape)
        self.L = Compartment(0., display_name="KL-Divergence", units="nats") 
        self.mu = Compartment(restVals, display_name="Predicted Probabilities (Q)") 
        self.dmu = Compartment(restVals) 
        
        self.target = Compartment(restVals, display_name="Target Distribution (P)") 
        self.dtarget = Compartment(restVals) 
        self.modulator = Compartment(restVals + 1.0) 
        self.mask = Compartment(restVals + 1.0)

    @staticmethod
    def eval_kl_div(target, mu, eps=1e-8):
        """
        Computes D_KL(P || Q) = sum( P * log(P/Q) )
        """
        P = jnp.clip(target, eps, 1.0)
        Q = jnp.clip(mu, eps, 1.0)
        kl = jnp.sum(P * (jnp.log(P) - jnp.log(Q)), axis=-1)
        return kl

    @compilable
    def advance_state(self, dt): 
        # Get variables
        mu = self.mu.get() # These are probabilities (0 to 1)
        target = self.target.get() # One-hot or target distribution
        modulator = self.modulator.get()
        mask = self.mask.get()

        # 1. Stable KL-Divergence Calculation for L
        # We use a slightly larger epsilon to prevent PPL from exploding
        eps = 1e-12 
        mu_safe = jnp.clip(mu, eps, 1.0)
        target_safe = jnp.clip(target, eps, 1.0)
        
        # D_KL(P || Q) = sum( P * (log P - log Q) )
        L_dist = jnp.sum(target * (jnp.log(target_safe) - jnp.log(mu_safe)), axis=-1)
        L = jnp.mean(L_dist) # Total loss for the batch

        # 2. Stable Error Signal (The "Mismatch")
        # Instead of the raw derivative -target/mu, we use (mu - target).
        # This is the 'Natural Gradient' and is numerically stable.
        dmu = (mu - target) 
        
        dtarget = -dmu # Symmetric error for the target compartment

        # Apply modulation and masking
        dmu = dmu * modulator * mask
        dtarget = dtarget * modulator * mask
        
        # Reset mask
        mask = mask * 0. + 1. 

        # Update compartments
        self.dmu.set(dmu)
        self.dtarget.set(dtarget)
        self.L.set(jnp.squeeze(L))
        self.mask.set(mask)

        
    @compilable
    def reset(self): 
        _shape = (self.batch_size, self.shape[0])
        if len(self.shape) > 1:
            _shape = (self.batch_size, self.shape[0], self.shape[1], self.shape[2])
        
        restVals = jnp.zeros(_shape)
        self.dmu.set(restVals)
        self.dtarget.set(restVals)
        self.target.set(restVals)
        self.mu.set(restVals)
        self.modulator.set(restVals + 1.0)
        self.L.set(0.)
        self.mask.set(jnp.ones(_shape))

    @classmethod
    def help(cls): 
        info = {
            cls.__name__: "Computes KL-Divergence for categorical distributions.",
            "compartments": {
                "mu": "Predicted probabilities (output of Softmax)",
                "target": "True distribution (one-hot or soft labels)",
                "L": "Scalar KL-Divergence value"
            },
            "formula": "sum( target * log(target / mu) )"
        }
        return info

if __name__ == '__main__':
    from ngcsimlib.context import Context
    with Context("ModelContext") as ctx:
        # Setup for a 10-class classification problem
        error_cell = CategoricalErrorCell("error_layer", n_units=10)
    print(error_cell)