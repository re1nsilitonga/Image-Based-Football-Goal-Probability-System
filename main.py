import cv2
import numpy as np
import sys
import os
from model import GoalProbabilityCalculator
from utils import ImageMarker, FieldProcessor, draw_visualizations

# Program utama untuk menghitung Probabilitas Gol berbasis gambar

def main():
    print("=== Goal Probability Calculator ===")
    
    # Pengaturan default untuk jalur gambar
    shot_img_path = os.path.join("input", "shot.jpg")
    field_img_path = os.path.join(os.getcwd(), "FIFAfield.png")
    
    # Memeriksa apakah file gambar default ada, jika tidak, meminta pengguna untuk memasukkan jalur file
    if not os.path.exists(shot_img_path):
        print(f"Warning: Default '{shot_img_path}' tidak ditemukan.")
        shot_img_path = input("Masukkan jalur ke Gambar Tembakan: ").strip('"')
    
    if not os.path.exists(field_img_path):
        print(f"Warning: Default '{field_img_path}' tidak ditemukan.")
        field_img_path = input("Masukkan jalur ke Gambar Lapangan FIFA: ").strip('"')
        
    marker = ImageMarker()
    processor = FieldProcessor()
    gp_calc = GoalProbabilityCalculator()
    
    # Langkah 1: Menandai gambar tembakan
    print("\n--- Langkah 1: Tandai Gambar Tembakan ---")
    shot_points = marker.mark_points(shot_img_path, [
        "Bola", 
        "Kiper", 
        "Tiang Atas-Kiri", 
        "Tiang Atas-Kanan", 
        "Tiang Bawah-Kanan", 
        "Tiang Bawah-Kiri"
    ])
    
    if not shot_points:
        print("Penandaan dibatalkan.")
        return

    # Langkah 2: Menandai posisi pemain bertahan pada gambar tembakan
    print("\n--- Langkah 2: Tandai Pemain Bertahan pada Gambar Tembakan ---")
    defender_points_shot = marker.mark_defenders(shot_img_path, shot_points)
    print(f"Menandai {len(defender_points_shot)} pemain bertahan.")
    if "Kiper" in shot_points:
        keeper_pt = shot_points["Kiper"]
        if keeper_pt not in defender_points_shot:
            defender_points_shot.append(keeper_pt)
            print("Menambahkan Kiper sebagai penghalang.")

    # Langkah 3: Menandai gambar lapangan FIFA
    print("\n--- Langkah 3: Tandai Lapangan FIFA ---")
    print("Silakan tandai titik-titik yang sesuai pada peta 2D.")
    field_points = marker.mark_points(field_img_path, [
        "Bola", 
        "Tiang Kiri", 
        "Tiang Kanan"
    ])
    
    if not field_points:
        print("Penandaan dibatalkan.")
        return

    # Langkah 4: Memproses data lapangan (Skala & Ground Truth)
    # Lebar gawang adalah 7.32m
    # Dalam gambar lapangan, Tiang Kiri dan Tiang Kanan sesuai dengan tiang asli
    left_post_field = field_points["Tiang Kiri"]
    right_post_field = field_points["Tiang Kanan"]
    
    pixels_per_meter = processor.calculate_scale(left_post_field, right_post_field)
    print(f"\nSkala: {pixels_per_meter:.2f} piksel/meter")
    
    # Menghitung jarak ground truth (Bola ke Tengah Gawang)
    ball_field = np.array(field_points["Bola"])
    goal_center_field = (np.array(left_post_field) + np.array(right_post_field)) / 2
    
    dist_pixels = np.linalg.norm(ball_field - goal_center_field)
    dist_meters = dist_pixels / pixels_per_meter
    
    # Menghitung sudut ground truth (Sudut terlihat dari mulut gawang)
    # Vektor dari Bola ke Tiang
    vec_l = np.array(left_post_field) - ball_field
    vec_r = np.array(right_post_field) - ball_field
    
    # Sudut menggunakan dot product
    dot_product = np.dot(vec_l, vec_r)
    norm_l = np.linalg.norm(vec_l)
    norm_r = np.linalg.norm(vec_r)
    
    if norm_l * norm_r == 0:
        angle_rad = 0
    else:
        cos_theta = np.clip(dot_product / (norm_l * norm_r), -1.0, 1.0)
        angle_rad = np.arccos(cos_theta)
    
    print(f"Jarak Ground Truth: {dist_meters:.2f} meter")
    print(f"Sudut Ground Truth: {np.degrees(angle_rad):.2f} derajat")
    
    # Langkah 5: Pemetaan Pemain Bertahan dengan Perubahan Basis Afine
    shot_origin = shot_points["Tiang Bawah-Kiri"]
    shot_a_point = shot_points["Tiang Bawah-Kanan"]
    shot_b_point = shot_points["Bola"]
    field_origin = field_points["Tiang Kiri"]
    field_a_point = field_points["Tiang Kanan"]
    field_b_point = field_points["Bola"]
    defenders_field = processor.transform_points_affine(
        defender_points_shot,
        shot_origin, shot_a_point, shot_b_point,
        field_origin, field_a_point, field_b_point
    )
    
    # Mengonversi posisi pemain bertahan ke koordinat relatif (meter relatif terhadap bola)
    direction_vector = goal_center_field - ball_field
    dir_norm = np.linalg.norm(direction_vector)
    
    if dir_norm == 0:
        a_unit = np.array([1.0, 0.0])
    else:
        a_unit = direction_vector / dir_norm
    
    defenders_relative = []
    for d_field in defenders_field:
        vec = d_field - ball_field
        d_x = np.dot(vec, a_unit) / pixels_per_meter
        d_y = abs((vec[0]*direction_vector[1] - vec[1]*direction_vector[0])) / (dir_norm * pixels_per_meter)
        defenders_relative.append(np.array([d_x, d_y]))
        
    print(f"Pemain Bertahan (Meter Relatif): {defenders_relative}")
    
    # Langkah 6: Menghitung Probabilitas Gol
    print("\n--- Langkah 4: Perhitungan Probabilitas Gol ---")
    
    res_standard = gp_calc.calculate_final_probability(dist_meters, angle_rad, defenders_relative, method='standard')
    print(f"\n[Metode Standar]")
    print(f"Probabilitas Dasar: {res_standard['base_probability']:.4f}")
    print(f"Faktor Halangan: {res_standard['obstacle_factor']:.4f}")
    print(f"Probabilitas Akhir: {res_standard['final_probability']:.4f}")
    
    defenders_vectors = []
    vec_goal_m = (goal_center_field - ball_field) / pixels_per_meter
    for d_field in defenders_field:
        vec_def_m = (d_field - ball_field) / pixels_per_meter
        defenders_vectors.append((vec_def_m, vec_goal_m))
    res_wedge = gp_calc.calculate_final_probability_with_wedge(dist_meters, angle_rad, defenders_vectors, method='standard')
    print(f"\n[Metode Wedge Product]")
    print(f"Probabilitas Dasar: {res_wedge['base_probability']:.4f}")
    print(f"Faktor Halangan: {res_wedge['obstacle_factor']:.4f}")
    print(f"Probabilitas Akhir: {res_wedge['final_probability']:.4f}")
    
    # Analisis Detail
    print("\n--- Analisis Detail ---")
    if not defenders_relative:
        print("Tidak ada halangan terdeteksi. Tembakan memiliki jalur jelas.")
    else:
        print(f"Jumlah Pemain Bertahan: {len(defenders_relative)}")
        # Memeriksa pemain bertahan terdekat
        dists = [np.linalg.norm(d) for d in defenders_relative]
        min_dist = min(dists)
        print(f"Pemain Bertahan Terdekat: {min_dist:.2f}m dari bola")
        
        # Memeriksa apakah ada pemain bertahan yang 'menghalangi'
        blockers = [d for d in defenders_relative if 0 < d[0] < dist_meters and abs(d[1]) < 1.0]
        if blockers:
            print(f"Halangan Langsung terdeteksi: {len(blockers)}")
        else:
            print("Tidak ada penghalang langsung pada jalur tembakan.")

    # Langkah 7: Visualisasi & Output
    print("\n--- Langkah 5: Visualisasi ---")
    final_prob_val = res_wedge['final_probability']
    
    # Memuat ulang gambar asli untuk menggambar pada kanvas bersih
    original_shot_img = cv2.imread(shot_img_path)
    
    # Menggambar hasil
    result_img = draw_visualizations(original_shot_img, shot_points, final_prob_val)
    
    # Menampilkan hasil
    display_result = marker._resize_to_screen(result_img)
    cv2.imshow("Hasil Akhir", display_result)
    
    # Menyimpan hasil
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, "output_" + os.path.basename(shot_img_path))
    cv2.imwrite(output_filename, result_img)
    print(f"Gambar hasil disimpan di: {output_filename}")
    
    print("\nTekan tombol apa saja pada jendela gambar untuk menutup...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\nSelesai.")

if __name__ == "__main__":
    main()
