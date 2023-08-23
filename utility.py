import cv2

def readim(dir, readtype):
    img = cv2.imread(dir, readtype)
    if img is None:
        raise RuntimeError(f"Could not read image at {dir}")
    else:
        return img