from multiprocessing import Process, Queue, Pipe
from multiprocessing.connection import Connection
from enum import Enum, auto

from gym import Env



class EnvStatus(Enum):
    RESET = auto()
    END = auto()



class ParallelEnv(Env):


    def __init__(self, envs, reset_args, reset_kwargs):
        super().__init__()
        self.envs = envs
        self.hub, self.remote = tuple(zip(*[Pipe(duplex=True) for _ in envs]))
        self.processes = [
            Process(target=runner, args=(env, remote, reset_args, reset_kwargs)) \
            for (env, remote) in zip(self.envs, self.remote)]
    

    def reset(self, *args, **kwargs):
        pass



def runner(env: Env, conn: Connection, reset_args: list=(), reset_kwargs: dict={}):
    while True:
        action = conn.recv()
        if action == EnvStatus.RESET:
            state = env.reset(*reset_args, **reset_kwargs)
            conn.send(state)
        elif action == EnvStatus.END:
            conn.close()
            break
        else:
            state, reward, done, _ = env.step(action)
            if done:
                state = env.reset(*reset_args, **reset_kwargs)
            conn.send((state, reward, done, _))
            