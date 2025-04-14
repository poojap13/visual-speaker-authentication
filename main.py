from src.generate_fakes import generate_fake_videos
from src.extract_optical_flow import process_real_dataset
import os

def main():
    # === Step 1: Generate DeepFake Videos ===
    print("\n[Step 1] Generating Fake Videos...")
    generate_fake_videos(
        base_dir="D:/visual_speaker_auth/data/gridcorpus",
        wav2lip_dir="D:/visual_speaker_auth/Wav2Lip",
        speaker_range=(1, 34),  # s1 to s33
        videos_per_user=500,
        step=2
    )

    # === Step 2: Extract Optical Flow from Real Videos ===
    print("\n[Step 2] Extracting Optical Flow (Real Videos)...")
    real_video_dir = "D:/visual_speaker_auth/data/gridcorpus/real_videos"
    real_flow_dir = "D:/visual_speaker_auth/data/gridcorpus/optical_flow/real"
    process_real_dataset(real_video_dir, real_flow_dir, target_size=(64, 64))

    # === Step 3: Extract Optical Flow from Fake Videos ===
    print("\n[Step 3] Extracting Optical Flow (Fake Videos)...")
    fake_video_dir = "D:/visual_speaker_auth/data/gridcorpus/fake_dataset"
    fake_flow_dir = "D:/visual_speaker_auth/data/gridcorpus/optical_flow/fake"
    process_real_dataset(fake_video_dir, fake_flow_dir, target_size=(64, 64))

    # === Step 4: Train MAML Model ===
    print("\n[Step 4] Training MAML Model...")
    os.system("python src/train_maml.py")

    # (Optional) === Step 5: Evaluate or Plot ===
    # os.system("python src/evaluate_maml.py")  # Add if needed

    print("\n✅ Pipeline complete. Models and results are ready.")

if __name__ == "__main__":
    main()
