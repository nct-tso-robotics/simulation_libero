"""Continue evaluation from a specific task and episode."""
import argparse
import datetime
import os

import cv2
import imageio
import numpy as np
import tqdm

from libero_policy_server import LiberoServer
from libero_socket_flags import LiberoRoutes

DATE_TIME = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def save_video(images, idx, success, task_desc):
    rollout_dir = f"./rollouts/{datetime.date.today()}"
    os.makedirs(rollout_dir, exist_ok=True)
    task_clean = task_desc.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={task_clean}.mp4"
    writer = imageio.get_writer(path, fps=30)
    for img in images:
        writer.append_data(img)
    writer.close()
    print(f"Saved: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Continue LIBERO evaluation from specific task/episode")
    parser.add_argument("--task", type=int, required=True, help="Task index to start from")
    parser.add_argument("--episode", type=int, required=True, help="Episode index to start from")
    parser.add_argument("--episodes_per_task", type=int, default=500, help="Total episodes per task")
    parser.add_argument("--num_tasks", type=int, default=10, help="Total number of tasks")
    parser.add_argument("--max_steps", type=int, default=520, help="Max steps per episode")
    parser.add_argument("--ip", type=str, default="0.0.0.0", help="Server IP")
    parser.add_argument("--port", type=int, default=5556, help="Server port")
    parser.add_argument("--suite", type=str, default="libero_10", help="Task suite name")
    # For resuming with existing counts
    parser.add_argument("--prev_episodes", type=int, default=0, help="Previously completed episodes (for SR calc)")
    parser.add_argument("--prev_successes", type=int, default=0, help="Previous successes (for SR calc)")
    args = parser.parse_args()

    server = LiberoServer(
        task_suite_name=args.suite,
        ip_address=args.ip,
        port=args.port,
        resolution=256,
        num_steps_wait=10,
        compression_type="raw",
    )

    print(f"Server ready on tcp://{args.ip}:{args.port}")
    print(f"Continuing from task {args.task}, episode {args.episode}")
    print("Waiting for client...")

    total_episodes = args.prev_episodes
    total_successes = args.prev_successes

    for task_idx in range(args.task, args.num_tasks):
        # Determine starting episode for this task
        start_ep = args.episode if task_idx == args.task else 0

        server._init_env_for_task(task_idx)
        task_desc = server.episode_state.task_description
        print(f"\n=== Task {task_idx}: {task_desc} ===")

        for ep_idx in tqdm.tqdm(range(start_ep, args.episodes_per_task), desc=f"Task {task_idx}"):
            print(f"\nEpisode {ep_idx + 1}/{args.episodes_per_task}")
            server._reset_episode(ep_idx)

            replay_images = []
            done = False
            t = 0

            while t < args.max_steps and not done:
                obs = server.current_obs
                agentview = obs.get("agentview_image")
                if agentview is not None:
                    if agentview.dtype != np.uint8:
                        agentview = (agentview * 255).astype(np.uint8)
                    replay_images.append(cv2.cvtColor(agentview, cv2.COLOR_BGR2RGB))

                route, done, success = server.handle_client_request()
                if route == LiberoRoutes.SEND_ACTION.value:
                    t += 1
                    if t % 20 == 0:
                        print(f"  Step {t}/{args.max_steps}")

            total_episodes += 1
            success = server.episode_state.success
            if success:
                total_successes += 1

            save_video(replay_images, total_episodes, success, task_desc)
            print(f"Success: {success} | Total: {total_episodes} | SR: {total_successes/total_episodes*100:.1f}%")

    server.shutdown()
    print(f"\n=== FINAL: {total_successes}/{total_episodes} = {total_successes/total_episodes*100:.1f}% ===")


if __name__ == "__main__":
    main()