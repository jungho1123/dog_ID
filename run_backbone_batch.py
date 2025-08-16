# run_backbone_batch.py
import yaml
import time
import shutil
import subprocess
from pathlib import Path

CONFIG_PATH = Path("config.yaml")
BACKBONES = [
    "tf_efficientnet_b4",
    "seresnext50_plain",
    "seresnext50_ibn_custom"
]



def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True)

def run_training(backbone_name):
    print(f"\n Start training: {backbone_name}")

    # config.yaml 수정
    cfg = load_config()
    cfg["model"]["name"] = backbone_name
    cfg["save"]["save_name"] = f"{backbone_name}.pth"
    save_config(cfg)

    # 별도 로그 경로 설정
    log_dir = Path("checkpoints") / backbone_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train_log.txt"

    # train.py 실행
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            ["python", "train.py"],
            stdout=log_file,
            stderr=subprocess.STDOUT
        )
        process.wait()

    print(f" Done: {backbone_name} → log saved to {log_path}")


def main():
    for name in BACKBONES:
        run_training(name)

    print("\n All training runs complete!")

if __name__ == "__main__":
    main()
