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
        # Max 3/4 of screen
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
            # Map back to original coordinates
            real_x = int(x / self.scale_factor)
            real_y = int(y / self.scale_factor)
            self.points.append((real_x, real_y))
            
            # Draw a circle on the display image (using display coordinates)
            cv2.circle(self.image, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow(self.window_name, self.image)

    def mark_points(self, image_path, point_names):
        """
        Opens an image and asks user to mark points in order.
        Returns a dictionary of point_name -> (x, y).
        """
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Resize for display
        display_img = self._resize_to_screen(original_image)
        self.image = display_img # Update reference for callback
        self.points = []

        cv2.imshow(self.window_name, display_img)
        cv2.setMouseCallback(self.window_name, self.click_event)

        print(f"Please mark the following points in order on the image window:")
        results = {}
        
        for name in point_names:
            print(f" - Mark: {name}")
            # Wait for a click
            current_len = len(self.points)
            while len(self.points) == current_len:
                key = cv2.waitKey(100) & 0xFF
                if key == 27: # Esc
                    cv2.destroyAllWindows()
                    return None
            
            # Store the last point (which is already in original coords)
            results[name] = self.points[-1]
            print(f"   Recorded {name} at {self.points[-1]}")
            
            # Draw label on display image
            # We need to calculate display coordinates for the label
            orig_pt = self.points[-1]
            disp_x = int(orig_pt[0] * self.scale_factor)
            disp_y = int(orig_pt[1] * self.scale_factor)
            
            cv2.putText(display_img, name, (disp_x + 10, disp_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow(self.window_name, display_img)

        # Allow marking defenders if needed
        # But for now, just return the fixed points
        # Wait a bit or wait for key
        print("Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return results

    def mark_defenders(self, image_path, existing_points=None):
        """
        Special mode to mark multiple defenders until a key is pressed.
        """
        # Reload or use existing image
        original_image = cv2.imread(image_path)
        display_img = self._resize_to_screen(original_image)
        self.image = display_img
        self.points = []
        
        # Draw existing points if any
        if existing_points:
            for name, pt in existing_points.items():
                # Convert to display coords
                disp_x = int(pt[0] * self.scale_factor)
                disp_y = int(pt[1] * self.scale_factor)
                
                cv2.circle(display_img, (disp_x, disp_y), 5, (0, 0, 255), -1)
                cv2.putText(display_img, name, (disp_x + 10, disp_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow(self.window_name, display_img)
        cv2.setMouseCallback(self.window_name, self.click_event)
        
        print("Mark Defenders. Press 'q' or 'Enter' when finished.")
        
        while True:
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q') or key == 13: # q or Enter
                break
        
        cv2.destroyAllWindows()
        return self.points

def draw_visualizations(image, points, xg_value):
    """
    Draws the xG visualization on the image.
    points: dict of point_name -> (x, y)
    xg_value: float (0.0 to 1.0)
    """
    img = image.copy()
    
    # Colors (BGR)
    COLOR_PINK = (255, 0, 255)
    COLOR_YELLOW = (0, 255, 255)
    COLOR_GREEN = (0, 255, 0)
    COLOR_BLACK = (0, 0, 0)
    
    # 1. Draw Goal Structure (if points exist)
    # Required: Goal Top-Left, Goal Top-Right, Goal Bottom-Right, Goal Bottom-Left
    required_goal = ["Goal Top-Left", "Goal Top-Right", "Goal Bottom-Right", "Goal Bottom-Left"]
    if all(k in points for k in required_goal):
        gtl = points["Goal Top-Left"]
        gtr = points["Goal Top-Right"]
        gbr = points["Goal Bottom-Right"]
        gbl = points["Goal Bottom-Left"]
        
        # Draw Goal Box
        cv2.line(img, gtl, gtr, COLOR_PINK, 2)
        cv2.line(img, gtr, gbr, COLOR_PINK, 2)
        cv2.line(img, gbr, gbl, COLOR_PINK, 2)
        cv2.line(img, gbl, gtl, COLOR_PINK, 2)
        
        # 2. Draw Triangle/Cone from Ball
        if "Ball" in points:
            ball = points["Ball"]
            cv2.line(img, ball, gbl, COLOR_PINK, 2)
            cv2.line(img, ball, gbr, COLOR_PINK, 2)
            
            # 3. Draw Trajectory (Yellow lines to corners or center)
            # Center of goal
            goal_center = ((gtl[0] + gbr[0]) // 2, (gtl[1] + gbr[1]) // 2)
            cv2.line(img, ball, goal_center, COLOR_YELLOW, 2)
            
            # Optional: Lines to top corners for "cone" effect
            cv2.line(img, ball, gtl, COLOR_YELLOW, 1)
            cv2.line(img, ball, gtr, COLOR_YELLOW, 1)

    # 4. Draw Overlay Text
    # "Goal Probability: XX.X%"
    text = f"Goal Probability: {xg_value * 100:.1f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0 # Slightly larger
    thickness = 2
    
    # Get text size
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Padding
    pad = 10
    # Box coordinates (Top-Left)
    box_x = 20
    box_y = 20
    box_w = text_w + 2 * pad
    box_h = text_h + 2 * pad
    
    # Draw Black Background Rectangle
    cv2.rectangle(img, (box_x, box_y), (box_x + box_w, box_y + box_h), COLOR_BLACK, -1)
    
    # Draw Green Text
    # Text origin is bottom-left of the text string
    text_x = box_x + pad
    text_y = box_y + pad + text_h
    cv2.putText(img, text, (text_x, text_y), font, font_scale, COLOR_GREEN, thickness)
    
    return img

class FieldProcessor:
    def __init__(self):
        self.real_goal_width = 7.32 # meters

    def calculate_scale(self, left_post, right_post):
        """
        Calculate pixels per meter based on goal posts.
        """
        dist_pixels = np.linalg.norm(np.array(left_post) - np.array(right_post))
        if dist_pixels == 0:
            return 1.0
        return dist_pixels / self.real_goal_width

    def compute_homography(self, src_points, dst_points):
        """
        Compute Homography matrix using Manual SVD (Singular Value Decomposition).
        Algorithms implemented manually:
        a) Sistem persamaan linier (Linear System construction)
        g) Dekomposisi matriks: SVD (Singular Value Decomposition)
        
        This replaces cv2.findHomography with a Direct Linear Transformation (DLT) approach.
        """
        src = np.array(src_points, dtype=np.float32)
        dst = np.array(dst_points, dtype=np.float32)
        
        if len(src) < 4 or len(dst) < 4:
            raise ValueError("Need at least 4 points for Homography")

        # 1. Construct Matrix A for the system Ah = 0
        # For each point correspondence (x,y) -> (xp,yp), we get 2 rows.
        # A is size (2*N, 9)
        A = []
        for i in range(len(src)):
            x, y = src[i][0], src[i][1]
            xp, yp = dst[i][0], dst[i][1]
            
            # Row 1: [-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp]
            A.append([-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp])
            # Row 2: [0, 0, 0, -x, -y, -1, x*yp, y*yp, yp]
            A.append([0, 0, 0, -x, -y, -1, x*yp, y*yp, yp])
            
        A = np.array(A)
        
        # 2. Solve using SVD (Decomposition)
        # A = U * Sigma * Vt
        # We use numpy's SVD which returns U, S, Vt
        U, S, Vt = np.linalg.svd(A)
        
        # The solution h is the eigenvector corresponding to the smallest eigenvalue (singular value)
        # In SVD, this is the last row of Vt (which corresponds to the smallest singular value in S)
        L = Vt[-1]
        
        # 3. Reshape into 3x3 Homography Matrix
        H = L.reshape(3, 3)
        
        # Normalize so H[2,2] = 1 (standard convention)
        if abs(H[2, 2]) > 1e-10:
            H = H / H[2, 2]
            
        return H

    def transform_points(self, points, H):
        """
        Transform a list of points using H.
        """
        if not points:
            return []
        pts = np.array([points], dtype=np.float32)
        dst = cv2.perspectiveTransform(pts, H)
        return dst[0]

    def transform_points_affine(self, points, src_origin, src_a_point, src_b_point, dst_origin, dst_a_point, dst_b_point):
        transformer = AffineBasisTransformer()
        return transformer.transform_points_affine(points, src_origin, src_a_point, src_b_point, dst_origin, dst_a_point, dst_b_point)

def wedge_product_2d(vec_a, vec_b):
    return vec_a[0] * vec_b[1] - vec_a[1] * vec_b[0]

class AffineBasisTransformer:
    def __init__(self):
        pass

    def _coefficients_change_of_basis(self, origin, a_point, b_point, p_point):
        o = np.array(origin, dtype=np.float64)
        a = np.array(a_point, dtype=np.float64) - o
        b = np.array(b_point, dtype=np.float64) - o
        p = np.array(p_point, dtype=np.float64) - o
        denom = wedge_product_2d(a, b)
        if abs(denom) < 1e-12:
            raise ValueError("Degenerate basis")
        alpha = wedge_product_2d(p, b) / denom
        beta = wedge_product_2d(p, a) / (-denom)
        return alpha, beta

    def transform_points_affine(self, points, src_origin, src_a_point, src_b_point, dst_origin, dst_a_point, dst_b_point):
        if not points:
            return []
        o_dst = np.array(dst_origin, dtype=np.float64)
        a_dst = np.array(dst_a_point, dtype=np.float64) - o_dst
        b_dst = np.array(dst_b_point, dtype=np.float64) - o_dst
        out = []
        for p in points:
            alpha, beta = self._coefficients_change_of_basis(src_origin, src_a_point, src_b_point, p)
            mapped = o_dst + alpha * a_dst + beta * b_dst
            out.append(mapped)
        return out

