import streamlit as st
import tensorflow as tf
import numpy as np
import json

from PIL import Image
from fpdf import FPDF
from datetime import datetime

from tensorflow.keras.applications.inception_v3 import preprocess_input

st.markdown("""
<style>

[data-testid="stAppViewContainer"]{
    background:
    radial-gradient(circle at top right,#14532d 0%,#020617 40%),
    linear-gradient(135deg,#020617,#052e16,#020617);
}

.block-container{
    padding-top:2rem;
}

.hero-title{
    font-size:64px;
    font-weight:800;
    color:white;
    line-height:1.1;
}

.hero-highlight{
    color:#22c55e;
}

.hero-sub{
    color:#cbd5e1;
    font-size:20px;
}

.glass{
    background:rgba(255,255,255,.05);
    backdrop-filter:blur(12px);
    border:1px solid rgba(255,255,255,.08);
    border-radius:24px;
    padding:25px;
}

.metric-card{
    background:rgba(255,255,255,.04);
    border-radius:18px;
    padding:20px;
    text-align:center;
    border:1px solid rgba(255,255,255,.06);
}

.result-card{
    background:rgba(34,197,94,.12);
    border-left:5px solid #22c55e;
    border-radius:20px;
    padding:20px;
}

.upload-box{
    padding:35px;
    text-align:center;
    border:2px dashed #22c55e;
    border-radius:20px;
    margin-bottom:20px;
}

</style>
""", unsafe_allow_html=True)

# 1. Judul dan Konfigurasi Halaman
st.set_page_config(
    page_title="TomatAI Premium",
    page_icon="🍅",
    layout="wide"
)

st.markdown("""
<style>

[data-testid="stFileUploader"]{
    background: rgba(255,255,255,0.04);
    border-radius:20px;
    padding:20px;
}

[data-testid="stFileUploader"] section{
    border:2px dashed #22c55e !important;
    border-radius:20px !important;
    background:transparent !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center;margin-top:20px'>

<div class='hero-title'>
🍅 TomatAI
<br>
<span class='hero-highlight'>
Deteksi & Solusi
</span>
<br>
Penyakit Daun Tomat
</div>

<br>

<div class='hero-sub'>
Unggah foto daun tomat untuk mengetahui penyakit
dan solusi penanganannya menggunakan AI.
</div>

</div>
""", unsafe_allow_html=True)

# 2. Database Solusi, Ciri-Ciri, dan Pencegahan (10 Kelas)
def dapatkan_informasi_penyakit(nama_kelas):
    database = {
        "Bercak_Bakteri": {
            "nama": "Bacterial Spot (Xanthomonas vesicatoria)",
            "ciri": "- Bercak kecil berair berwarna coklat tua-hitam\n- Tepi bercak berwarna kuning\n- Bercak pada daun, batang, dan buah\n- Buah berbintik kasar dan tidak layak jual",
            "pencegahan": "- Gunakan benih bebas patogen\n- Hindari penyiraman dari atas\n- Rotasi tanaman\n- Jangan bekerja di kebun saat daun basah",
            "solusi": "- Semprot bakterisida tembaga (copper hydroxide)\n- Kombinasi tembaga + mankozeb lebih efektif\n- Buang bagian tanaman yang terinfeksi parah"
        },
        "Hawar_Awal": {
            "nama": "Early Blight (Alternaria solani)",
            "ciri": "- Bercak coklat tua dengan pola lingkaran (target)\n- Dimulai dari daun tua bagian bawah\n- Tepi bercak berwarna kuning\n- Daun gugur prematur, batang bisa terinfeksi",
            "pencegahan": "- Rotasi tanaman minimal 2 tahun\n- Buang sisa tanaman sakit setelah panen\n- Hindari kelembapan berlebih\n- Pupuk nitrogen cukup (tidak berlebihan)",
            "solusi": "- Fungisida mankozeb, klorotalonil, atau iprodion\n- Semprot sejak gejala pertama muncul\n- Agens hayati: Bacillus subtilis"
        },
        "Hawar_Lambat": {
            "nama": "Late Blight (Phytophthora infestans)",
            "ciri": "- Bercak hijau keabu-abuan bertepi tidak beraturan\n- Sisi bawah daun: lapisan putih seperti kapas (sporangia)\n- Penyebaran sangat cepat dalam kondisi lembap\n- Seluruh tanaman bisa mati dalam hitungan hari",
            "pencegahan": "- Hindari kelembapan tinggi dan suhu 10-25°C terus-menerus\n- Gunakan varietas tahan P. infestans\n- Penyemprotan fungisida preventif di musim hujan\n- Jarak tanam lebar untuk sirkulasi udara",
            "solusi": "- Fungisida metalaksil, dimethomorph, atau cymoxanil\n- Semprot setiap 5-7 hari saat musim hujan\n- Cabut dan musnahkan tanaman terinfeksi parah"
        },
        "Kapang_Daun": {
            "nama": "Leaf Mold (Passalora fulva)",
            "ciri": "- Bercak kuning pucat di sisi atas daun\n- Sisi bawah daun: bulu-bulu abu-abu keunguan (koloni jamur)\n- Daun menggulung dan mengering\n- Umumnya menyerang di rumah kaca (kelembapan tinggi)",
            "pencegahan": "- Kurangi kelembapan udara di dalam greenhouse\n- Buka ventilasi untuk sirkulasi udara\n- Hindari percikan air ke daun\n- Gunakan varietas tahan Leaf Mold",
            "solusi": "- Fungisida berbahan aktif difenokonazol atau propikonazol\n- Buang daun terinfeksi\n- Kurangi irigasi dan tingkatkan ventilasi"
        },
        "Bercak_Septoria": {
            "nama": "Septoria Leaf Spot (Septoria lycopersici)",
            "ciri": "- Bercak kecil (2-4 mm) berwarna putih-abu dengan tepi coklat tua\n- Titik hitam kecil (piknidium) di tengah bercak\n- Dimulai dari daun bawah, naik ke atas\n- Daun menguning dan gugur, tanaman meranggas",
            "pencegahan": "- Rotasi tanaman 2-3 tahun\n- Mulsa tanah untuk cegah percikan spora\n- Buang daun bawah yang terinfeksi\n- Penyiraman di pangkal tanaman",
            "solusi": "- Fungisida mankozeb, klorotalonil, atau kaptan\n- Semprot setiap 7-10 hari\n- Aplikasi konsisten mulai awal musim"
        },
        "Tungau_Laba_Laba": {
            "nama": "Spider Mites / Tungau Merah (Tetranychus urticae)",
            "ciri": "- Titik-titik putih/keperakan kecil di permukaan atas daun\n- Jaring halus di sisi bawah daun (terutama cuaca panas)\n- Daun menguning, mengering, dan gugur\n- Serangan parah: tanaman tampak seperti terbakar",
            "pencegahan": "- Jaga kelembapan kebun (tungau suka kering)\n- Semprot air ke daun bawah secara berkala\n- Hindari penggunaan insektisida berlebihan (membunuh predator alami)\n- Tanam tanaman refugia sebagai habitat musuh alami",
            "solusi": "- Akarisida: abamektin, bifenazat, atau spirodifen\n- Sabun insektisida (insecticidal soap) untuk investasi ringan\n- Pelepasan musuh alami: Phytoseiulus persimilis"
        },
        "Bercak_Target": {
            "nama": "Target Spot (Corynespora cassiicola)",
            "ciri": "- Bercak coklat bulat seperti pola sasaran (target)\n- Lingkaran konsentris pada bercak\n- Daun menguning di sekitar bercak\n- Bercak membesar dan menyatu, daun gugur",
            "pencegahan": "- Rotasi tanaman 2-3 tahun sekali\n- Hindari penyiraman dari atas (gunakan irigasi tetes)\n- Jaga sirkulasi udara dengan jarak tanam cukup\n- Buang daun terinfeksi segera",
            "solusi": "- Fungisida berbahan aktif azoksistrobin atau mankozeb\n- Semprot pagi hari saat daun kering\n- Aplikasi Trichoderma sp. sebagai agens hayati"
        },
        "Virus_Menggulung_Kuning": {
            "nama": "Yellow Leaf Curl Virus / TYLCV (Begomovirus)",
            "ciri": "- Daun menguning dan menggulung ke atas\n- Daun mengecil dan menebal\n- Tanaman kerdil, produksi buah sangat berkurang\n- Ditularkan oleh kutu kebul (Bemisia tabaci)",
            "pencegahan": "- Pasang mulsa plastik perak untuk mengusir kutu kebul\n- Gunakan insektisida sistemik sejak dini\n- Tanam varietas tahan TYLCV\n- Pasang perangkap kuning (yellow sticky trap)",
            "solusi": "- Cabut tanaman bergejala berat\n- Semprot imidakloprid atau tiametoksam untuk vektor\n- Isolasi tanaman sakit dari tanaman sehat"
        },
        "Mosaic_Virus": {
            "nama": "Tomato Mosaic Virus / ToMV (Tobamovirus)",
            "ciri": "- Pola mosaik hijau terang-hijau tua pada daun\n- Daun berkerut dan terdistorsi\n- Pertumbuhan tanaman terhambat\n- Buah berbintik dan matang tidak merata",
            "pencegahan": "- Gunakan benih bersertifikat bebas virus\n- Desinfeksi alat pertanian sebelum digunakan\n- Kendalikan serangga vektor (kutu daun, thrips)\n- Tanam varietas tahan virus",
            "solusi": "- Cabut dan musnahkan tanaman terinfeksi\n- Tidak ada fungisida/pestisida efektif; fokus pada pencegahan\n- Semprot insektisida untuk kendalikan vektor"
        },
        "Sehat": {
            "nama": "Healthy / Sehat",
            "ciri": "- Daun hijau cerah dan rata\n- Tidak ada bercak, bintik, atau perubahan warna\n- Pertumbuhan normal dan seragam\n- Tekstur daun halus tanpa kerutan",
            "pencegahan": "- Lanjutkan praktik budidaya baik (GAP)\n- Pemupukan berimbang NPK + mikro\n- Monitoring rutin minimal seminggu sekali\n- Jaga sanitasi kebun",
            "solusi": "- Tidak diperlukan penanganan khusus\n- Pertahankan kondisi tanah, air, dan nutrisi optimal\n- Dokumentasikan kondisi sebagai referensi baseline"
        }
    }
    return database.get(nama_kelas, None)

# 3. Fungsi Generator Cetak PDF Resmi
def buat_pdf(info, confidence):
    pdf = FPDF()
    pdf.add_page()
    
    # Header Dokumen
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(200, 30, 30)
    pdf.cell(0, 10, "LAPORAN DIAGNOSIS PENYAKIT TANAMAN TOMAT", ln=True, align="C")
    
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(100, 100, 100)
    waktu_sekarang = datetime.now().strftime("%d %B %Y - %H:%M WIB")
    pdf.cell(0, 10, f"Dicetak otomatis oleh TomatAI pada: {waktu_sekarang}", ln=True, align="C")
    pdf.ln(5)
    
    # Garis Pembatas
    pdf.set_draw_color(150, 150, 150)
    pdf.line(10, 32, 200, 32)
    pdf.ln(5)
    
    # Hasil Analisis AI
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, "1. HASIL DIAGNOSIS SISTEM AI", ln=True)
    
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(40, 8, "Nama Penyakit:", ln=False)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, f"{info['nama']}", ln=True)
    
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(40, 8, "Tingkat Keyakinan:", ln=False)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, f"{confidence:.2f}%", ln=True)
    pdf.ln(5)
    
    # Detail Rekomendasi Medis Tanaman
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "2. DETAIL GEJALA DAN SOLUSI PENANGANAN", ln=True)
    
    # Menggunakan multi_cell agar teks panjang otomatis ganti baris dan tidak terpotong kertas
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(0, 102, 204) # Warna biru
    pdf.cell(0, 6, "A. Ciri-Ciri Gejala Lapangan:", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, 6, info['ciri'])
    pdf.ln(3)
    
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(204, 153, 0) # Warna kuning tua
    pdf.cell(0, 6, "B. Langkah Tindakan Pencegahan:", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, 6, info['pencegahan'])
    pdf.ln(3)
    
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(204, 0, 0) # Warna merah
    pdf.cell(0, 6, "C. Solusi Pengobatan Kuratif:", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, 6, info['solusi'])
    
    pdf.ln(15)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 5, "*Catatan: Laporan ini dihasilkan oleh kecerdasan buatan (AI) berbasis analisis citra daun.", ln=True, align="L")
    pdf.cell(0, 5, "Harap lakukan pemantauan berkala pada area kebun Anda.", ln=True, align="L")
    
    return bytes(pdf.output())

# 4. Load Model AI
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model("model_tomat_final.keras")

try:
    model = load_my_model()

    with open("class_names.json", "r") as f:
        class_names = json.load(f)

    label_mapping = {
        "Tomato_Bacterial_spot": "Bercak_Bakteri",
        "Tomato_Early_blight": "Hawar_Awal",
        "Tomato_Late_blight": "Hawar_Lambat",
        "Tomato_Leaf_Mold": "Kapang_Daun",
        "Tomato_Septoria_leaf_spot": "Bercak_Septoria",
        "Tomato_Spider_mites_Two_spotted_spider_mite": "Tungau_Laba_Laba",
        "Tomato__Target_Spot": "Bercak_Target",
        "Tomato__Tomato_YellowLeaf__Curl_Virus": "Virus_Menggulung_Kuning",
        "Tomato__Tomato_mosaic_virus": "Mosaic_Virus",
        "Tomato_healthy": "Sehat"
    }

except Exception as e:
    st.error(f"Gagal memuat model AI: {e}")

col1,col2,col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class='metric-card'>
    <h3>🎯 Akurasi Tinggi</h3>
    <p>Model CNN mendeteksi 10 penyakit daun tomat.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='metric-card'>
    <h3>⚡ Cepat</h3>
    <p>Analisis hanya beberapa detik.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='metric-card'>
    <h3>📋 Solusi Lengkap</h3>
    <p>Gejala, pencegahan dan pengobatan otomatis.</p>
    </div>
    """, unsafe_allow_html=True)
    
# 5. Fitur Upload Gambar
st.markdown("""
<div style='text-align:center;padding:40px;'>

<h1 style='font-size:56px;color:white;'>
📤 Unggah Foto Daun Tomat
</h1>

<p style='font-size:22px;color:#cbd5e1;'>
Pilih gambar JPG atau PNG yang jelas
untuk dianalisis oleh AI
</p>

</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "",
    type=["jpg","jpeg","png"],
    label_visibility="collapsed"
)


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang Diunggah", use_container_width=True)
    
    with st.spinner("AI sedang mendiagnosis daun..."):
        image = image.convert("RGB")

        img_resized = image.resize((224, 224))

        img_array = np.array(img_resized)

        img_input = np.expand_dims(
                    img_array,
                    axis=0
                )
        img_array = preprocess_input(img_array)

        predictions = model.predict(
            img_input,
            verbose=0
        )

        class_id = np.argmax(predictions[0])

        predicted_class = class_names[class_id]

        predicted_class = label_mapping[predicted_class]

        confidence = float(
            np.max(predictions[0]) * 100
        )        
        
        info = dapatkan_informasi_penyakit(predicted_class)
        
        # Tampilan Output Utama
        left,right = st.columns([1,1])

        with left:
            st.image(
                image,
                caption="Daun Tomat",
                use_container_width=True
            )

        with right:

            st.markdown(f"""
            <div class='result-card'>

            <h2>🔍 Hasil Diagnosis</h2>

            <h3>{info['nama']}</h3>

            <h4>Confidence: {confidence:.2f}%</h4>

            </div>
            """, unsafe_allow_html=True)

            st.progress(int(confidence))

            st.metric(
                "Tingkat Keyakinan AI",
                f"{confidence:.2f}%"
            )
        
        st.markdown("---")
        st.subheader("📋 Detail Laporan & Penanganan Medis:")
        
        tab1,tab2,tab3 = st.tabs([
            "👁️ Gejala",
            "🛡️ Pencegahan",
            "💊 Solusi"
        ])

        with tab1:
            st.markdown(info["ciri"])

        with tab2:
            st.markdown(info["pencegahan"])

        with tab3:
            st.markdown(info["solusi"])
        
        st.markdown("---")
        # 6. TOMBOL EXPORT PDF OTOMATIS
        st.subheader("🖨️ Cetak PDF Hasil Analisa:")
        
        # Generate biner file PDF
        pdf_data = buat_pdf(info, confidence)
        
        # Menampilkan tombol download bawaan Streamlit
        st.download_button(
            "📥 Download Laporan PDF",
            pdf_data,
            file_name=f"Laporan_{predicted_class}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

st.markdown("""
<hr>

<center>

🍅 TomatAI Premium

AI-Based Tomato Leaf Disease Detection System

© 2026

</center>
""", unsafe_allow_html=True)
