from PIL import Image
import io
import numpy as np

def validasi_daun_tomat(image_bytes: bytes, media_type: str = "image/jpeg") -> dict:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(image.resize((100, 100)))
    
    r = img_array[:,:,0].astype(float)
    g = img_array[:,:,1].astype(float)
    b = img_array[:,:,2].astype(float)
    
    # Hitung piksel yang benar-benar hijau daun
    # Hijau daun: g jauh lebih tinggi dari r dan b, dan cukup gelap (bukan putih)
    hijau_daun = (
        (g > r + 15) &      # hijau lebih dominan dari merah
        (g > b + 15) &      # hijau lebih dominan dari biru
        (g > 40) &          # bukan terlalu gelap
        (g < 220)           # bukan putih/terang banget
    )
    
    persen_hijau = hijau_daun.sum() / (100 * 100) * 100
    
    # Minimal 15% piksel harus hijau daun
    if persen_hijau >= 15:
        return {"is_valid": True, "reason": f"Gambar terdeteksi sebagai daun tomat ({persen_hijau:.1f}% hijau)."}
    else:
        return {"is_valid": False, "reason": f"Gambar bukan daun tomat ({persen_hijau:.1f}% hijau). Silakan upload gambar daun tomat yang jelas."}