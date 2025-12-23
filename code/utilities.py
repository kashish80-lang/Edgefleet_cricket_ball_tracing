

def interpolate_points(p1, p2, num_points):
    
    x1, y1 = p1
    x2, y2 = p2
    points = []
    for i in range(num_points):
        t = i / max(num_points-1, 1)
        x = int(x1 + (x2 - x1) * t)
        y = int(y1 + (y2 - y1) * t)
        points.append((x, y))
    return points

def draw_trajectory(frame, trajectory, color=(0,0,255), thickness=2):
    
    for i in range(1, len(trajectory)):
        cv2.line(frame, trajectory[i-1], trajectory[i], color, thickness)
