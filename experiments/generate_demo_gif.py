"""
Generate a GIF demonstration of trained agent playing Ludo.
Captures frames from visual gameplay and saves as animated GIF.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import cv2
import imageio
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rl_agent_ludo.environment.level5_multiagent import Level5MultiAgentLudo
from rl_agent_ludo.agents.simple_dqn import SimpleDQNAgent
from rl_agent_ludo.environment.standard_board_visualizer import StandardLudoBoardVisualizer


def extract_positions_level5(env):
    """Extract positions for Level 5 (4 players, 2 tokens each)."""
    positions = [env.my_tokens.copy()]
    for opp_tokens in env.opponent_tokens:
        positions.append(opp_tokens.copy())
    return positions


def generate_gameplay_gif(
    checkpoint_path='checkpoints/level5/best_model.pth',
    output_path='assets/demo_gameplay.gif',
    num_episodes=1,
    max_steps=300,
    fps=10,
    resize_factor=0.7
):
    """
    Generate a GIF of agent playing Ludo.

    Args:
        checkpoint_path: Path to trained model
        output_path: Where to save the GIF
        num_episodes: Number of game episodes to record
        max_steps: Maximum steps per episode (to keep GIF size manageable)
        fps: Frames per second in output GIF
        resize_factor: Factor to resize frames (reduce file size)
    """
    print("="*80)
    print("GENERATING DEMO GIF")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_path}")
    print(f"Episodes: {num_episodes}")
    print(f"Max steps per episode: {max_steps}")
    print(f"FPS: {fps}")
    print("="*80 + "\n")

    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load environment and agent
    print("Loading environment and agent...")
    env = Level5MultiAgentLudo(seed=42)

    agent = SimpleDQNAgent(
        state_dim=16,
        action_dim=3,
        hidden_dims=[128, 128],
        device='cpu'
    )

    # Check if checkpoint exists
    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Using random agent instead...")
        use_random = True
    else:
        agent.load(checkpoint_path)
        use_random = False
        print("Agent loaded successfully!")

    # Create visualizer
    viz = StandardLudoBoardVisualizer(
        level=5,
        num_players=4,
        tokens_per_player=2
    )

    frames = []
    total_frames = 0

    print(f"\nRecording gameplay...")

    try:
        for ep in range(num_episodes):
            print(f"\nEpisode {ep+1}/{num_episodes}")
            obs, info = env.reset(seed=42 + ep)
            done = False
            steps = 0

            pbar = tqdm(total=max_steps, desc=f"Episode {ep+1}")

            while not done and steps < max_steps:
                # Extract current state for visualization
                positions = extract_positions_level5(env)

                state_info = {
                    'player_positions': positions,
                    'current_player': 0,  # Always player 0 for Level 5
                    'dice': env.current_dice,
                    'winner': None,  # Will be set at end
                    'step': steps
                }

                # Render and capture frame
                board_original = viz._draw_board()
                viz._draw_header_panel(board_original, state_info)
                viz._draw_all_pieces(board_original, state_info['player_positions'])
                viz._draw_legend(board_original)
                viz._draw_status(board_original, state_info)

                # Copy the frame
                frame = board_original.copy()

                # Also show it (optional)
                cv2.imshow(viz.window_name, board_original)
                cv2.waitKey(1)

                # Resize to reduce file size
                if resize_factor != 1.0:
                    new_width = int(frame.shape[1] * resize_factor)
                    new_height = int(frame.shape[0] * resize_factor)
                    frame = cv2.resize(frame, (new_width, new_height))

                # Convert BGR to RGB for imageio
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                total_frames += 1

                # Agent acts
                if use_random:
                    action = np.random.choice([0, 1, 2])
                else:
                    action = agent.act(obs, greedy=True)

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1

                pbar.update(1)

                # Every N frames, show progress
                if steps % 50 == 0:
                    pbar.set_postfix({'frames': total_frames})

            pbar.close()

            # Capture final state
            positions = extract_positions_level5(env)
            final_winner = info.get('winner', -1)
            state_info = {
                'player_positions': positions,
                'current_player': 0,
                'dice': env.current_dice,
                'winner': final_winner,
                'step': steps
            }

            # Render final frame
            board_final = viz._draw_board()
            viz._draw_header_panel(board_final, state_info)
            viz._draw_all_pieces(board_final, state_info['player_positions'])
            viz._draw_legend(board_final)
            viz._draw_status(board_final, state_info)
            frame = board_final.copy()

            # Show it
            cv2.imshow(viz.window_name, board_final)
            cv2.waitKey(1)

            if resize_factor != 1.0:
                new_width = int(frame.shape[1] * resize_factor)
                new_height = int(frame.shape[0] * resize_factor)
                frame = cv2.resize(frame, (new_width, new_height))

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Add final frame multiple times for pause effect
            for _ in range(int(fps * 2)):  # 2 second pause
                frames.append(frame_rgb)
                total_frames += 1

            winner = info.get('winner', -1)
            result = "WON! 🎉" if winner == 0 else "LOST"
            print(f"Episode {ep+1} finished: {result} in {steps} steps")

    finally:
        viz.close()

    # Save as GIF
    print(f"\nSaving GIF with {total_frames} frames...")
    print(f"Output: {output_path}")

    # Calculate duration per frame
    duration = 1.0 / fps

    imageio.mimsave(
        output_path,
        frames,
        duration=duration,
        loop=0  # Infinite loop
    )

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n{'='*80}")
    print(f"GIF GENERATION COMPLETE!")
    print(f"{'='*80}")
    print(f"Output: {output_path}")
    print(f"Total frames: {total_frames}")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"Duration: {total_frames / fps:.1f} seconds")
    print(f"{'='*80}")

    return output_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Generate demo GIF of agent playing')
    parser.add_argument('--checkpoint', type=str,
                       default='checkpoints/level5/best_model.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--output', type=str,
                       default='assets/demo_gameplay.gif',
                       help='Output GIF path')
    parser.add_argument('--episodes', type=int, default=1,
                       help='Number of episodes to record')
    parser.add_argument('--max_steps', type=int, default=250,
                       help='Max steps per episode')
    parser.add_argument('--fps', type=int, default=10,
                       help='Frames per second')
    parser.add_argument('--resize', type=float, default=0.7,
                       help='Resize factor (0.5 = half size, 1.0 = full size)')

    args = parser.parse_args()

    generate_gameplay_gif(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        fps=args.fps,
        resize_factor=args.resize
    )


if __name__ == "__main__":
    main()
