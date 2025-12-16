# tracker.py
import numpy as np

class Tracker:
    def __init__(self):
        self.next_id = 0
        self.objects = {}  # object_id: centroid

    def update(self, detections):
        updated_objects = {}
        for det in detections:
            x, y, w, h = det
            cx, cy = x + w // 2, y + h // 2

            # Find closest existing object
            min_dist = float('inf')
            obj_id = None
            for oid, (ox, oy) in self.objects.items():
                dist = np.hypot(cx - ox, cy - oy)
                if dist < min_dist:
                    min_dist = dist
                    obj_id = oid

            if min_dist < 50:  # distance threshold
                updated_objects[obj_id] = (cx, cy)
            else:
                updated_objects[self.next_id] = (cx, cy)
                self.next_id += 1

        self.objects = updated_objects
        return self.objects