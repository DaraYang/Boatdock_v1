from gym.envs.registration import register
from pamdp_env import boatEnv

register(
    id='boatdock-v0',
    entry_point = 'pamdp_env:boatEnv',
    kwargs={'choice': 1, 'testmode':1, 'km': None}
)
