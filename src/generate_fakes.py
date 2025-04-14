import os
import subprocess

def generate_fake_videos(
    base_dir="D:/visual_speaker_auth/data/gridcorpus",
    wav2lip_dir="D:/visual_speaker_auth/Wav2Lip",
    speaker_range=(1, 34),
    videos_per_user=500,
    step=2
):
    users = [f"s{i}" for i in range(speaker_range[0], speaker_range[1])]
    output_base_dir = os.path.join(base_dir, "fake_dataset")
    checkpoint_path = os.path.join(wav2lip_dir, "checkpoints/wav2lip.pth")

    os.makedirs(output_base_dir, exist_ok=True)

    for user in users:
        print(f"Processing user: {user}")
        video_dir = os.path.join(base_dir, "video", user)
        audio_dir = os.path.join(base_dir, "audio", user)
        output_dir = os.path.join(output_base_dir, user)
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(video_dir) or not os.path.exists(audio_dir):
            print(f"Missing directory for user {user}. Skipping...")
            continue

        video_files = sorted([f for f in os.listdir(video_dir) if f.endswith(('.mpg', '.mp4'))])
        selected_videos = video_files[::step][:videos_per_user]

        for video_file in selected_videos:
            video_path = os.path.join(video_dir, video_file)
            audio_file = os.path.splitext(video_file)[0] + '.wav'
            audio_path = os.path.join(audio_dir, audio_file)

            if not os.path.exists(audio_path):
                print(f"Audio file {audio_file} not found for video {video_file}. Skipping...")
                continue

            output_file = f'fake_{os.path.splitext(video_file)[0]}.avi'
            output_path = os.path.join(output_dir, output_file)

            command = [
                "python", os.path.join(wav2lip_dir, "inference.py"),
                "--checkpoint_path", checkpoint_path,
                "--face", video_path,
                "--audio", audio_path,
                "--outfile", output_path
            ]

            print(f"Generating fake video for {video_file}...")
            result = subprocess.run(command, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"Error processing {video_file} with {audio_file}:\n{result.stderr}")
            else:
                print(f"Successfully generated {output_file}")

        print(f"Completed processing for user: {user}\n")


if __name__ == "__main__":
    generate_fake_videos()
