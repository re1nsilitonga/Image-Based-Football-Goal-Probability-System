import numpy as np

class GoalProbabilityCalculator:
    def __init__(self):
        self.b_intercept = 2.0  
        self.b_distance = -0.15
        self.b_angle = 1.2
        self.keeper_weight = 1.8

    def calculate_base_probability(self, distance, angle):
        logit = self.b_intercept + (self.b_distance * distance) + (self.b_angle * angle)
        prob = 1 / (1 + np.exp(-logit))
        return prob

    def calculate_final_probability(self, distance, angle, defenders, method='standard', roles=None):
        base_prob = self.calculate_base_probability(distance, angle)
        
        if not defenders:
            obstacle_factor = 1.0
        else:
            if method == 'eigenvalue':
                obstacle_factor = self._calculate_obstacle_factor_with_eigenvalue(distance, angle, defenders)
            else:
                obstacle_factor = self._calculate_obstacle_factor(distance, angle, defenders, roles)
        
        final_prob = base_prob * obstacle_factor
        return {
            "base_probability": base_prob,
            "obstacle_factor": obstacle_factor,
            "final_probability": final_prob,
            "method": method
        }

    def _calculate_individual_scores(self, distance, defenders, roles=None):
        scores = []
        for i, d in enumerate(defenders):
            d_x, d_y = d # x adalah jarak ke depan, y adalah offset lateral
            
            if 0 < d_x < distance:
                sigma = 0.5 # meter
                effectiveness = np.exp(-(d_y**2) / (2 * sigma**2))
                
                angle_blocked = np.arctan(0.5 / d_x) # asumsi lebar pemain disamakan 0.5m
                
                score = effectiveness * (angle_blocked * 2)
                if roles is not None and i < len(roles) and roles[i] == 'keeper':
                    score *= self.keeper_weight
                scores.append(score)
            else:
                scores.append(0.0)
        return scores

    def _calculate_obstacle_factor(self, distance, angle, defenders, roles=None):
        scores = self._calculate_individual_scores(distance, defenders, roles)
        blocking_score = sum(scores)
        
        factor = np.exp(-1.0 * blocking_score)
        return factor

    def _calculate_eigen_manual_2x2(self, matrix):
        a = matrix[0, 0]
        b = matrix[0, 1]
        c = matrix[1, 0]
        d = matrix[1, 1]
        
        tr = a + d
        det = a*d - b*c
        
        delta = tr**2 - 4*det
        
        if delta < 0:
            return np.array([]), np.array([])
            
        sqrt_delta = np.sqrt(delta)
        l1 = (tr + sqrt_delta) / 2
        l2 = (tr - sqrt_delta) / 2
        
        vals = [l1, l2]
        
        vecs = []
        for lam in vals:
            if abs(b) > 1e-10:
                v = np.array([1, -(a - lam)/b])
            elif abs(c) > 1e-10:
                v = np.array([-(d - lam)/c, 1])
            else:
                if abs(a - lam) < 1e-10:
                    v = np.array([1, 0])
                else:
                    v = np.array([0, 1])
            
            norm = np.sqrt(v[0]**2 + v[1]**2)
            if norm > 0:
                vecs.append(v / norm)
            else:
                vecs.append(v)
            
        vecs_mat = np.column_stack(vecs)
        return np.array(vals), vecs_mat

    def _calculate_obstacle_factor_with_eigenvalue(self, distance, angle, defenders):
        if len(defenders) < 2:
            return self._calculate_obstacle_factor(distance, angle, defenders)

        points = np.array(defenders)
        
        mean_pos = np.mean(points, axis=0)
        centered_points = points - mean_pos
        
        N = centered_points.shape[0]
        cov_matrix = np.dot(centered_points.T, centered_points) / (N - 1)
        
        try:
            eigenvalues, eigenvectors = self._calculate_eigen_manual_2x2(cov_matrix)
            if len(eigenvalues) == 0:
                 eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        except:
            return self._calculate_obstacle_factor(distance, angle, defenders)
            
        idx = eigenvalues.argsort()[::-1]   
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:,idx]
        
        v1 = eigenvectors[:, 0]
        angle_v1 = np.arctan2(v1[1], v1[0])
        
        angle_diff = abs(angle_v1)
        if angle_diff > np.pi:
            angle_diff -= np.pi
        if angle_diff > np.pi / 2:
            angle_diff = np.pi - angle_diff
            
        wallness = np.sin(angle_diff)
        
        scores = self._calculate_individual_scores(distance, defenders)
        sum_score = sum(scores)
        max_score = max(scores) if scores else 0
        
        effective_score = (wallness * sum_score) + ((1.0 - wallness) * max_score)
        
        factor = np.exp(-1.0 * effective_score)
        return factor

    def _calculate_individual_scores_wedge(self, defenders_vectors, roles=None):
        scores = []
        for i, (vec_def, vec_goal) in enumerate(defenders_vectors):
            dist_goal = np.linalg.norm(vec_goal)
            if dist_goal == 0:
                scores.append(0.0)
                continue
            d_x = np.dot(vec_def, vec_goal) / dist_goal
            wedge_val = vec_def[0]*vec_goal[1] - vec_def[1]*vec_goal[0]
            d_y = abs(wedge_val) / dist_goal
            if 0 < d_x < dist_goal:
                sigma = 0.5
                effectiveness = np.exp(-(d_y**2) / (2 * sigma**2))
                angle_blocked = np.arctan(0.5 / d_x)
                score = effectiveness * (angle_blocked * 2)
                if roles is not None and i < len(roles) and roles[i] == 'keeper':
                    score *= self.keeper_weight
                scores.append(score)
            else:
                scores.append(0.0)
        return scores

    def calculate_final_probability_with_wedge(self, distance, angle, defenders_vectors, method='standard', roles=None):
        base_prob = self.calculate_base_probability(distance, angle)
        if not defenders_vectors:
            obstacle_factor = 1.0
        else:
            scores = self._calculate_individual_scores_wedge(defenders_vectors, roles)
            blocking_score = sum(scores)
            obstacle_factor = np.exp(-1.0 * blocking_score)
        final_prob = base_prob * obstacle_factor
        return {
            "base_probability": base_prob,
            "obstacle_factor": obstacle_factor,
            "final_probability": final_prob,
            "method": "wedge_product_ga"
        }
