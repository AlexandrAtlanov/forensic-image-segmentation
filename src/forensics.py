import cv2
import numpy as np

ELA_JPEG_QUALITY = 90


def compute_ela(img_rgb: np.ndarray, quality: int = ELA_JPEG_QUALITY) -> np.ndarray:
    ok, buf = cv2.imencode('.jpg', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError(f'кодирование не удалось для картинки {img_rgb.shape}')
    
    re = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    diff = np.abs(img_rgb.astype(np.float32) - re.astype(np.float32)).mean(axis=2)

    return np.clip(diff / 255.0, 0.0, 1.0)


def compute_highpass(img_rgb: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(img_rgb, (3, 3), 0)
    diff = np.abs(img_rgb.astype(np.float32) - blur.astype(np.float32)).mean(axis=2)
    
    return np.clip(diff / 255.0, 0.0, 1.0)