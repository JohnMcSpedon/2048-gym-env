from .env_2048 import Env2048

import gymnasium

gymnasium.register(id="Gym2048-v0", entry_point=Env2048)
