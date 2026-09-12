"""Joint adversarial motion prior and denoising world model PPO."""

from src.rsl_rl.algorithms.amp_ppo import AMPPPO
from src.rsl_rl.algorithms.dwl_ppo import DWLPPO


class AMPDWLPPO(DWLPPO, AMPPPO):
    """Combine DWL recurrent representation learning with AMP style rewards."""

    pass
