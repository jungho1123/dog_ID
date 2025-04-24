# utils/output_saver.py

from pathlib import Path

def get_output_dir(cfg):
    """
    모델 저장 디렉토리 하위에 결과물 저장 폴더 생성
    ex) checkpoints/seresnext50_ibn_custom/
    """
    model_name = Path(cfg["save"]["save_name"]).stem
    out_dir = Path(cfg["save"]["save_dir"]) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def get_output_path(cfg, filename):
    """
    prefix (raw_/norm_)를 붙인 파일 경로 반환
    ex) checkpoints/seresnext50_ibn_custom/raw_umap2d.png
    """
    prefix = cfg.get("visualization", {}).get("prefix", "")
    return get_output_dir(cfg) / f"{prefix}{filename}"
