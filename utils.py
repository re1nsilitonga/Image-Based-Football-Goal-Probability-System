import cv2
import numpy as np
try:
    import tkinter as tk
except ImportError:
    tk = None

class ImageMarker:
    def __init__(self):
        self.points = []
        self.image = None
        self.window_name = "Mark Points"
        self.scale_factor = 1.0

    def _get_screen_size(self):
        if tk:
            try:
                root = tk.Tk()
                root.withdraw()
                w, h = root.winfo_screenwidth(), root.winfo_screenheight()
                root.destroy()
                return w, h
            except Exception:
                pass
        return 1920, 1080 # Default fallback

    def _resize_to_screen(self, image):
        screen_w, screen_h = self._get_screen_size()
        # Maksimal 3/4 dari layar biar pengguna mudah saat marker
        max_w = int(screen_w * 0.75)
        max_h = int(screen_h * 0.75)
        
        h, w = image.shape[:2]
        scale = 1.0
        
        if w > max_w or h > max_h:
            scale_w = max_w / w
            scale_h = max_h / h
            scale = min(scale_w, scale_h)
            
        self.scale_factor = scale
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            return cv2.resize(image, (new_w, new_h))
        return image.copy()

    def click_event(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Memetakan kembali ke koordinat asli
            real_x = int(x / self.scale_factor)
            real_y = int(y / self.scale_factor)
            self.points.append((real_x, real_y))
            
            # Menggambar lingkaran pada gambar tampilan (menggunakan koordinat tampilan)
            cv2.circle(self.image, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow(self.window_name, self.image)

    def mark_points(self, image_path, point_names):
        """
        Membuka gambar dan meminta pengguna untuk menandai titik secara berurutan.
        Mengembalikan dictionary point_name -> (x, y).
        """
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"Tidak dapat memuat gambar: {image_path}")
        
        # Ubah ukuran untuk tampilan
        display_img = self._resize_to_screen(original_image)
        self.image = display_img # Perbarui referensi untuk callback
        self.points = []

        cv2.imshow(self.window_name, display_img)
        cv2.setMouseCallback(self.window_name, self.click_event)

        print(f"Silakan tandai titik-titik berikut secara berurutan pada jendela gambar:")
        results = {}
        
        for name in point_names:
            print(f" - Tandai: {name}")
            current_len = len(self.points)
            while len(self.points) == current_len:
                key = cv2.waitKey(100) & 0xFF
                if key == 27: # Esc
                    cv2.destroyAllWindows()
                    return None
            
            # Simpan titik terakhir (yang sudah dalam koordinat asli)
            results[name] = self.points[-1]
            print(f"   Tercatat {name} di {self.points[-1]}")
            
            # Menggambar label pada gambar tampilan
            # Menghitung koordinat tampilan untuk label
            orig_pt = self.points[-1]
            disp_x = int(orig_pt[0] * self.scale_factor)
            disp_y = int(orig_pt[1] * self.scale_factor)
            
            cv2.putText(display_img, name, (disp_x + 10, disp_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow(self.window_name, display_img)

        # Izinkan penandaan penghalang jika diperlukan
        print("Tekan tombol apa saja untuk melanjutkan...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return results

    def mark_defenders(self, image_path, existing_points=None):
        """
        Mode khusus untuk menandai beberapa penghalang hingga tombol ditekan.
        """
        # Muat ulang atau gunakan gambar yang ada
        original_image = cv2.imread(image_path)
        display_img = self._resize_to_screen(original_image)
        self.image = display_img
        self.points = []
        
        # Gambar titik yang ada jika ada
        if existing_points:
            for name, pt in existing_points.items():
                # Konversi ke koordinat tampilan
                disp_x = int(pt[0] * self.scale_factor)
                disp_y = int(pt[1] * self.scale_factor)
                
                cv2.circle(display_img, (disp_x, disp_y), 5, (0, 0, 255), -1)
                cv2.putText(display_img, name, (disp_x + 10, disp_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow(self.window_name, display_img)
        cv2.setMouseCallback(self.window_name, self.click_event)
        
        print("Tandai Penghalang. Tekan 'q' atau 'Enter' jika selesai.")
        
        while True:
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q') or key == 13:
                break
        
        cv2.destroyAllWindows()
        return self.points

def draw_visualizations(image, points, goal_probability):
    """
    Menggambar visualisasi Probabilitas Gol pada gambar.
    points: dict dari point_name -> (x, y)
    goal_probability: float (0.0 hingga 1.0)
    """
    img = image.copy()
    
    # Warna (BGR)
    COLOR_PINK = (255, 0, 255)
    COLOR_YELLOW = (0, 255, 255)
    COLOR_GREEN = (0, 255, 0)
    COLOR_BLACK = (0, 0, 0)
    
    # 1. Gambar Struktur Gawang (jika titik ada)
    # Dibutuhkan: Goal Top-Left, Goal Top-Right, Goal Bottom-Right, Goal Bottom-Left
    required_goal = ["Goal Top-Left", "Goal Top-Right", "Goal Bottom-Right", "Goal Bottom-Left"]
    if all(k in points for k in required_goal):
        gtl = points["Goal Top-Left"]
        gtr = points["Goal Top-Right"]
        gbr = points["Goal Bottom-Right"]
        gbl = points["Goal Bottom-Left"]
        
        # Gambar Kotak Gawang
        cv2.line(img, gtl, gtr, COLOR_PINK, 2)
        cv2.line(img, gtr, gbr, COLOR_PINK, 2)
        cv2.line(img, gbr, gbl, COLOR_PINK, 2)
        cv2.line(img, gbl, gtl, COLOR_PINK, 2)
        
        # 2. Gambar Segitiga/Kerucut dari Bola
        if "Ball" in points:
            ball = points["Ball"]
            cv2.line(img, ball, gbl, COLOR_PINK, 2)
            cv2.line(img, ball, gbr, COLOR_PINK, 2)
            
            # 3. Gambar Trajektori (Garis Kuning ke sudut atau tengah)
            # Tengah gawang
            goal_center = ((gtl[0] + gbr[0]) // 2, (gtl[1] + gbr[1]) // 2)
            cv2.line(img, ball, goal_center, COLOR_YELLOW, 2)
            
            # Opsional: Garis ke sudut atas untuk efek "kerucut"
            cv2.line(img, ball, gtl, COLOR_YELLOW, 1)
            cv2.line(img, ball, gtr, COLOR_YELLOW, 1)

    # 4. Gambar Teks Overlay
    text = f"Probabilitas Gol: {goal_probability * 100:.1f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0 # Sedikit lebih besar
    thickness = 2
    
    # Dapatkan ukuran teks
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Padding
    pad = 10
    # Koordinat Kotak (Kiri-Atas)
    box_x = 20
    box_y = 20
    box_w = text_w + 2 * pad
    box_h = text_h + 2 * pad
    
    # Gambar Persegi Panjang Latar Belakang Hitam
    cv2.rectangle(img, (box_x, box_y), (box_x + box_w, box_y + box_h), COLOR_BLACK, -1)
    
    # Gambar Teks Hijau
    # Asal teks adalah kiri-bawah string teks
    text_x = box_x + pad
    text_y = box_y + pad + text_h
    cv2.putText(img, text, (text_x, text_y), font, font_scale, COLOR_GREEN, thickness)
    
    return img

class FieldProcessor:
    def __init__(self):
        self.real_goal_width = 7.32 # meter berdasarkan regulasi FIFA

    def calculate_scale(self, left_post, right_post):
        """
        Menghitung piksel per meter berdasarkan tiang gawang.
        """
        dist_pixels = np.linalg.norm(np.array(left_post) - np.array(right_post))
        if dist_pixels == 0:
            return 1.0
        return dist_pixels / self.real_goal_width

    def compute_homography(self, src_points, dst_points):
        """
        Menghitung matriks Homografi menggunakan SVD Manual (Singular Value Decomposition).
        Algoritma diimplementasikan secara manual:
        a) Sistem persamaan linier (Konstruksi Sistem Linier)
        g) Dekomposisi matriks: SVD (Singular Value Decomposition)
        
        Ini menggantikan cv2.findHomography dengan pendekatan Direct Linear Transformation (DLT).
        """
        src = np.array(src_points, dtype=np.float32)
        dst = np.array(dst_points, dtype=np.float32)
        
        if len(src) < 4 or len(dst) < 4:
            raise ValueError("Butuh setidaknya 4 titik untuk Homografi")

        # 1. Konstruksi Matriks A untuk sistem Ah = 0
        # Untuk setiap korespondensi titik (x,y) -> (xp,yp), kita mendapatkan 2 baris.
        # A berukuran (2*N, 9)
        A = []
        for i in range(len(src)):
            x, y = src[i][0], src[i][1]
            xp, yp = dst[i][0], dst[i][1]
            
            # Baris 1: [-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp]
            A.append([-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp])
            # Baris 2: [0, 0, 0, -x, -y, -1, x*yp, y*yp, yp]
            A.append([0, 0, 0, -x, -y, -1, x*yp, y*yp, yp])
            
        A = np.array(A)
        
        # 2. Selesaikan menggunakan SVD (Dekomposisi)
        # A = U * Sigma * Vt
        # Kita menggunakan SVD numpy yang mengembalikan U, S, Vt
        U, S, Vt = np.linalg.svd(A)
        
        # Solusi h adalah vektor eigen yang sesuai dengan nilai eigen terkecil (nilai singular)
        # Dalam SVD, ini adalah baris terakhir dari Vt (yang sesuai dengan nilai singular terkecil di S)
        L = Vt[-1]
        
        # 3. Bentuk ulang menjadi Matriks Homografi 3x3
        H = L.reshape(3, 3)
        
        # Normalisasi sehingga H[2,2] = 1 (konvensi standar)
        if abs(H[2, 2]) > 1e-10:
            H = H / H[2, 2]
            
        return H

    def transform_points(self, points, H):
        """
        Transformasi daftar titik menggunakan H.
        """
        if not points:
            return []
        pts = np.array([points], dtype=np.float32)
        dst = cv2.perspectiveTransform(pts, H)
        return dst[0]

