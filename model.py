import numpy as np

class XGCalculator:
    def __init__(self):
        # Koefisien untuk model xG sederhana (Logistic Regression)
        # xG = 1 / (1 + exp(-(Intercept + b_dist * jarak + b_angle * sudut)))
        # Intercept dicocokkan agar mencapai probabilitas ~75% untuk situasi penalti
        self.b_intercept = 2.0  
        self.b_distance = -0.15 # probabilitas menurun seiring bertambahnya jarak
        self.b_angle = 1.2      # probabilitas meningkat seiring bertambahnya sudut (dalam radian)

    def calculate_base_xg(self, distance, angle):
        """
        Menghitung xG dasar berdasarkan jarak (meter) dan sudut (radian).
        """
        logit = self.b_intercept + (self.b_distance * distance) + (self.b_angle * angle)
        xg = 1 / (1 + np.exp(-logit))
        return xg

    def calculate_final_xg(self, distance, angle, defenders, method='standard'):
        """
        Menghitung xG akhir dengan mempertimbangkan penghalang.
        """
        base_xg = self.calculate_base_xg(distance, angle)
        
        if not defenders:
            # Skor default 1.0 (pengali) jika tidak ada penghalang
            obstacle_factor = 1.0
        else:
            if method == 'eigenvalue':
                obstacle_factor = self._calculate_obstacle_factor_with_eigenvalue(distance, angle, defenders)
            else:
                obstacle_factor = self._calculate_obstacle_factor(distance, angle, defenders)
        
        final_xg = base_xg * obstacle_factor
        return {
            "base_xg": base_xg,
            "obstacle_factor": obstacle_factor,
            "final_xg": final_xg,
            "method": method
        }

    def _calculate_individual_scores(self, distance, defenders):
        scores = []
        for d in defenders:
            d_x, d_y = d # x adalah jarak ke depan, y adalah offset lateral
            
            # Periksa apakah penghalang berada di antara bola dan gawang
            if 0 < d_x < distance:
                # Efektivitas tergantung pada offset lateral
                sigma = 0.5 # meter
                effectiveness = np.exp(-(d_y**2) / (2 * sigma**2))
                
                # Juga tergantung pada jarak dari penembak (lebih dekat = lebih banyak blok)
                # Sudut yang diblokir ~ lebar / d_x
                angle_blocked = np.arctan(0.5 / d_x) # asumsi lebar pemain disamakan 0.5m
                
                score = effectiveness * (angle_blocked * 2)
                scores.append(score)
            else:
                scores.append(0.0)
        return scores

    def _calculate_obstacle_factor(self, distance, angle, defenders):
        """
        Perhitungan standar: Jumlah efektivitas blok.
        """
        scores = self._calculate_individual_scores(distance, defenders)
        blocking_score = sum(scores)
        
        # Faktor adalah pengali [0, 1]
        factor = np.exp(-1.0 * blocking_score)
        return factor

    def _calculate_eigen_manual_2x2(self, matrix):
        """
        Perhitungan manual Nilai Eigen dan Vektor Eigen untuk matriks 2x2.
        Algoritma yang diimplementasikan secara manual:
        b) Determinan (Perhitungan Determinan)
        e) Nilai eigen dan vektor eigen (Persamaan Karakteristik)
        """
        a = matrix[0, 0]
        b = matrix[0, 1]
        c = matrix[1, 0]
        d = matrix[1, 1]
        
        # 1. Jejak dan Determinan
        # Determinan = ad - bc
        tr = a + d
        det = a*d - b*c
        
        # 2. Persamaan Karakteristik: lambda^2 - tr*lambda + det = 0
        delta = tr**2 - 4*det
        
        if delta < 0:
            return np.array([]), np.array([])
            
        sqrt_delta = np.sqrt(delta)
        l1 = (tr + sqrt_delta) / 2
        l2 = (tr - sqrt_delta) / 2
        
        vals = [l1, l2]
        
        # 3. Vektor Eigen
        # Menyelesaikan (A - lambda*I)v = 0
        vecs = []
        for lam in vals:
            # Baris 1: (a - lam)x + by = 0
            if abs(b) > 1e-10:
                v = np.array([1, -(a - lam)/b])
            elif abs(c) > 1e-10:
                v = np.array([-(d - lam)/c, 1])
            else:
                # Matriks diagonal
                if abs(a - lam) < 1e-10:
                    v = np.array([1, 0])
                else:
                    v = np.array([0, 1])
            
            # Normalisasi vektor (Norma Euclidean)
            norm = np.sqrt(v[0]**2 + v[1]**2)
            if norm > 0:
                vecs.append(v / norm)
            else:
                vecs.append(v)
            
        # Format mirip dengan np.linalg.eig: vals, vecs (kolom-wise)
        vecs_mat = np.column_stack(vecs)
        return np.array(vals), vecs_mat

    def _calculate_obstacle_factor_with_eigenvalue(self, distance, angle, defenders):
        """
        Analisis Nilai Eigen: Menganalisis bentuk pertahanan.
        """
        if len(defenders) < 2:
            return self._calculate_obstacle_factor(distance, angle, defenders)

        # Konversi ke array numpy
        points = np.array(defenders)
        
        # Menghitung Matriks Kovarian dari posisi penghalang
        # Memusatkan titik-titik
        mean_pos = np.mean(points, axis=0)
        centered_points = points - mean_pos
        
        # Perhitungan Kovarian Manual
        # Cov = (X^T * X) / (N - 1)
        N = centered_points.shape[0]
        cov_matrix = np.dot(centered_points.T, centered_points) / (N - 1)
        
        # Nilai Eigen
        try:
            eigenvalues, eigenvectors = self._calculate_eigen_manual_2x2(cov_matrix)
            if len(eigenvalues) == 0:
                 eigenvalues, eigenvectors = np.linalg.eig(cov_matrix) # Cadangan
        except:
            return self._calculate_obstacle_factor(distance, angle, defenders)
            
        # Urutkan nilai eigen
        idx = eigenvalues.argsort()[::-1]   
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:,idx]
        
        # Identifikasi Arah Utama
        v1 = eigenvectors[:, 0]
        angle_v1 = np.arctan2(v1[1], v1[0])
        
        # Perbedaan dari sudut tembakan (0 radian sepanjang sumbu X)
        angle_diff = abs(angle_v1)
        # Normalisasi ke [0, pi/2]
        if angle_diff > np.pi:
            angle_diff -= np.pi
        if angle_diff > np.pi / 2:
            angle_diff = np.pi - angle_diff
            
        # "Wallness": 1 jika tegak lurus (pi/2), 0 jika sejajar (0)
        wallness = np.sin(angle_diff)
        
        # Menghitung Skor
        scores = self._calculate_individual_scores(distance, defenders)
        sum_score = sum(scores)
        max_score = max(scores) if scores else 0
        
        # Interpolasi Skor Efektif
        effective_score = (wallness * sum_score) + ((1.0 - wallness) * max_score)
        
        factor = np.exp(-1.0 * effective_score)
        return factor
