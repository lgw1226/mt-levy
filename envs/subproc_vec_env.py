'''
    Adapted from Stable Baseline 3 https://github.com/DLR-RM/stable-baselines3/tree/master
'''
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection

import gymnasium as gym
import numpy as np
import cloudpickle
from wrappers import MTWrapper


class CloudpickleWrapper:
    def __init__(self, var):
        """
        Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)

        :param var: (Any) the variable you wish to wrap for pickling with cloudpickle
        """
        self.var = var

    def __getstate__(self):
        return cloudpickle.dumps(self.var)

    def __setstate__(self, var):
        self.var = cloudpickle.loads(var)


def _worker(conn: Connection, work_conn: Connection, serialized_env_fn: CloudpickleWrapper, seed: int):
    """Create a worker function that will be used by a `multiprocessing.Process` instance.

    :param Connection conn: Connection object used to communicate with the parent process.
    :param Connection conn: Connection object used to communicate with the worker process.
    """
    conn.close()  # close the parent connection in the subprocess (no need to access main process connection)

    # TODO: seeding, rendering, and other functionalities are not yet supported
    env: gym.Env = serialized_env_fn.var()
    if isinstance(env, MTWrapper):
        env.unwrapped.seed(seed)
    while True:
        try:
            cmd, data = work_conn.recv()
            if cmd == 'step':
                observation, reward, terminated, truncated, info = env.step(data)
                work_conn.send((observation, reward, terminated, truncated, info))
            elif cmd == 'reset':
                observation, info = env.reset()
                work_conn.send((observation, info))
            elif cmd == 'render':
                work_conn.send(env.render())
            elif cmd == 'get_attr':
                work_conn.send(getattr(env, data))
            elif cmd == 'stop':
                work_conn.close()
                break
            elif cmd == 'close':
                env.close()
                work_conn.close()
                break
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break

class SubprocVecEnv:
    """
    Creates a multiprocess vectorized wrapper for multiple environments, distributing each environment to its own
    process, allowing significant speed up when the environment is computationally complex.

    For performance reasons, if your environment is not IO bound, the number of environments should not exceed the
    number of logical cores on your CPU.

    :param List[Callable[[], gym.Env]] env_fns: A list of functions that will create the environments
        (each callable returns a `Gym.Env` instance when called).
    """

    def __init__(self, env_fns, seed: int):
        self.waiting = False
        self.closed = False
        self.num_envs = len(env_fns)

        conn_pairs: tuple[tuple[Connection], tuple[Connection]] = zip(*[Pipe() for _ in range(self.num_envs)])  # create pipelines for communication between parent and subprocesses
        self.conns, self.work_conns = conn_pairs
        self.processes: list[Process] = []
        for rank, (conn, work_conn, env_fn) in enumerate(zip(self.conns, self.work_conns, env_fns)):
            args = (conn, work_conn, CloudpickleWrapper(env_fn), seed+rank)
            process = Process(target=_worker, args=args, daemon=True)  # create subprocess instance
            process.start()
            self.processes.append(process)
            work_conn.close()  # close the worker connection in the parent process (no need to access subprocess connection)

    def _step_async(self, actions: tuple[np.ndarray]):
        for conn, action in zip(self.conns, actions):
            conn.send(('step', action))
        self.waiting = True

    def _step_wait(self) -> tuple[tuple[np.ndarray], tuple[float], tuple[bool], tuple[bool], tuple[dict]]:
        results = [conn.recv() for conn in self.conns]
        self.waiting = False
        obs, rwd, ter, tru, info = zip(*results)
        return obs, rwd, ter, tru, info
    
    def step(self, actions: tuple[np.ndarray]) -> tuple[tuple[np.ndarray], tuple[float], tuple[bool], tuple[bool], tuple[dict]]:
        self._step_async(actions)
        return self._step_wait()
    
    def sample_action(self) -> tuple[np.ndarray]:
        for conn in self.conns:
            conn.send(('get_attr', 'action_space'))
        results = [conn.recv() for conn in self.conns]
        return tuple([space.sample() for space in results])

    def reset(self) -> tuple[np.ndarray, tuple[dict]]:
        for conn in self.conns:
            conn.send(('reset', None))
        results = [conn.recv() for conn in self.conns]
        obs, info = zip(*results)  # Unpacking correctly
        return obs, info
    
    def stop(self):
        for conn in self.conns:
            conn.send(('stop', None))

    def render(self) -> tuple[np.ndarray]:
        '''Return tuple of rendered images.'''
        for conn in self.conns:
            conn.send(('render', None))
        return tuple([conn.recv() for conn in self.conns])

    def close(self):
        if self.closed:
            return

        if self.waiting:
            try:
                for conn in self.conns:
                    if conn.poll():  # Only receive if a message is available
                        conn.recv()
            except (EOFError, BrokenPipeError):
                pass  # Avoid blocking if process is already closed

        for conn in self.conns:
            conn.send(('stop', None))

        for process in self.processes:
            process.join(timeout=1)  # Avoid infinite blocking

        self.closed = True