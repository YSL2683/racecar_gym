"""
Evaluation server: runs MultiAgentRaceEnv and serves observations to remote clients.

Protocol (per message, using pickle over TCP):
  Client → Server  {'agent_id': str}
  Server → Client  {'type': 'reset',  'obs': ..., 'state': ..., 'episode': int}
  Client → Server  {'action': dict}
  Server → Client  {'type': 'step',   'obs': ..., 'reward': float,
                                       'done': bool, 'info': dict}
  Client → Server  {'action': dict}   (repeat until done=True — client skips action when done)
  Server → Client  {'type': 'close'}  (after all episodes, or on client failure)

Episode semantics:
  An episode ends when *any* agent's termination condition is met (any(dones)).
  Both clients receive the same global done=True at that step.

Usage:
  python server_client/server.py --scenario scenarios/eval_austria.yml --episodes 2 --render-mode human
"""

import argparse
import queue
import socket
import sys
import os
import threading
from typing import Dict
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import racecar_gym.envs.gym_api  # noqa: F401 — registers gymnasium environments
from racecar_gym.envs.gym_api import MultiAgentRaceEnv
from server_client.utils import send_msg, recv_msg

# How long (seconds) the main loop waits for an action before declaring failure.
ACTION_TIMEOUT = 30.0
# Socket-level read timeout for client connections.
SOCKET_TIMEOUT = 60.0
TIME_STEP = 0.01

def run_server(
    scenario: str,
    host: str,
    port: int,
    num_episodes: int,
    render_mode: str,
    reset_mode: str,
) -> None:
    env = MultiAgentRaceEnv(scenario=scenario, render_mode=render_mode)
    agent_ids = list(env.scenario.agents.keys())
    num_agents = len(agent_ids)

    print(f"Scenario : {scenario}")
    print(f"Agents   : {agent_ids}")
    print(f"Episodes : {num_episodes}  |  reset_mode: {reset_mode}\n")

    # Per-agent queues: main simulation thread <-> client I/O threads.
    action_queues: Dict[str, queue.Queue] = {aid: queue.Queue(1) for aid in agent_ids}
    result_queues: Dict[str, queue.Queue] = {aid: queue.Queue(1) for aid in agent_ids}

    # Set when any client I/O thread encounters a fatal error.
    failure_event = threading.Event()

    # ── Accept connections ────────────────────────────────────────────────────
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((host, port))
    server_sock.listen(num_agents)
    print(f"Listening on {host}:{port}  —  waiting for {num_agents} client(s)...")

    connections: Dict[str, socket.socket] = {}
    for _ in range(num_agents):
        conn, addr = server_sock.accept()
        conn.settimeout(SOCKET_TIMEOUT)
        msg = recv_msg(conn)
        if msg is None or 'agent_id' not in msg:
            raise ConnectionError(f"Invalid handshake from {addr}")
        aid = msg['agent_id']
        if aid not in agent_ids:
            conn.close()
            raise ValueError(f"Unknown agent_id '{aid}'. Valid IDs: {agent_ids}")
        if aid in connections:
            conn.close()
            raise ValueError(f"Duplicate connection for agent_id '{aid}'")
        connections[aid] = conn
        print(f"  Agent '{aid}' connected from {addr}")

    print("All clients connected.\n")

    # ── Per-client I/O threads ────────────────────────────────────────────────
    def client_worker(agent_id: str, conn: socket.socket) -> None:
        """Relay between socket and the agent's queues.

        Loop alternates:
          1. Dequeue a result and send it to the client.
          2. Unless terminal (done / close), receive an action and enqueue it.

        On any failure, sets failure_event so the main loop can abort cleanly.
        """
        try:
            while True:
                result = result_queues[agent_id].get()
                send_msg(conn, result)

                if result['type'] == 'close':
                    break

                # Receive next action only when the episode is still running.
                needs_action = result['type'] == 'reset' or not result.get('done', False)
                if needs_action:
                    msg = recv_msg(conn)
                    if msg is None:
                        raise ConnectionError(f"Agent '{agent_id}' disconnected.")
                    action_queues[agent_id].put(msg['action'])

        except Exception as e:
            print(f"[!] client_worker error ({agent_id}): {e}")
            failure_event.set()

    threads = [
        threading.Thread(target=client_worker, args=(aid, connections[aid]), daemon=True)
        for aid in agent_ids
    ]
    for t in threads:
        t.start()

    # ── Main simulation loop ──────────────────────────────────────────────────
    def get_action(agent_id: str) -> object:
        """Get an action from the queue, respecting the failure_event."""
        while True:
            try:
                return action_queues[agent_id].get(timeout=1.0)
            except queue.Empty:
                if failure_event.is_set():
                    raise RuntimeError(f"Client failure detected while waiting for '{agent_id}'.")

    aborted = False
    for episode in range(num_episodes):
        if failure_event.is_set():
            print("[!] Client failure — aborting remaining episodes.")
            aborted = True
            break

        obs, state = env.reset(options={'mode': reset_mode})

        for aid in agent_ids:
            result_queues[aid].put({
                'type': 'reset',
                'obs': obs[aid],
                'state': state[aid],
                'episode': episode,
            })

        step = 0
        try:
            while True:
                start_time = time.time()

                actions = {aid: get_action(aid) for aid in agent_ids}

                obs, rewards, dones, _, state = env.step(actions)
                # Episode ends when any agent's termination condition is met.
                episode_done = any(dones.values())

                if render_mode != 'human':
                    frame = env.render()
                    if frame is not None:
                        bgr = cv2.cvtColor(frame.astype('uint8'), cv2.COLOR_RGB2BGR)
                        cv2.imshow('racecar_gym', bgr)
                        cv2.waitKey(1)

                for aid in agent_ids:
                    result_queues[aid].put({
                        'type': 'step',
                        'obs': obs[aid],
                        'reward': rewards[aid],
                        'done': episode_done,
                        'info': state[aid],
                    })

                step += 1
                elapsed_time = time.time() - start_time
                sleep_time = max(0, TIME_STEP - elapsed_time)
                time.sleep(sleep_time)
                if episode_done:
                    break

            print(f"Episode {episode + 1}/{num_episodes} finished in {step} steps.")

        except RuntimeError as e:
            print(f"[!] {e}")
            aborted = True
            break

    # Signal all clients that the session is over.
    for aid in agent_ids:
        try:
            result_queues[aid].put({'type': 'close'}, timeout=2.0)
        except queue.Full:
            pass

    for t in threads:
        t.join(timeout=5.0)

    for conn in connections.values():
        conn.close()
    server_sock.close()
    env.close()
    if render_mode != 'human':
        cv2.destroyAllWindows()

    if aborted:
        print("\nServer closed (session aborted due to client failure).")
    else:
        print("\nServer closed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Racecar Gym evaluation server')
    parser.add_argument('--scenario', type=str, required=True,
                        help='Path to scenario YAML (e.g. scenarios/eval_austria.yml)')
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=5555)
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--render-mode', type=str, default='human',
                        choices=['human', 'rgb_array_follow', 'rgb_array_birds_eye'])
    parser.add_argument('--reset-mode', type=str, default='grid',
                        choices=['grid', 'random', 'random_bidirectional'])
    args = parser.parse_args()

    run_server(
        scenario=args.scenario,
        host=args.host,
        port=args.port,
        num_episodes=args.episodes,
        render_mode=args.render_mode,
        reset_mode=args.reset_mode,
    )
