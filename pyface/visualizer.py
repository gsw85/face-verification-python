
import cv2

class FaceVisualizer:

    FONT_TYPE     = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE    = 0.8
    FONT_COLOR    = (255, 0, 0) # BGR
    BOX_COLOR     = (255, 255, 255) # BGR
    BOX_THICKNESS = 2

    def show_boxes(self, img, bounding_boxes):
        for box in bounding_boxes:
            cv2.rectangle(
                img,
                (box['xmin'], box['ymin']),
                (box['xmax'], box['ymax']),
                self.BOX_COLOR,
                self.BOX_THICKNESS
            )


    def show_landmarks(self, img, landmark_points):
        # loop over facial landmarks
        for coords in landmark_points:
            # loop over the (x, y)-coordinates for the facial landmarks
            for (x, y) in coords:
                cv2.circle(img, (x, y), 1, self.BOX_COLOR, thickness=-1)


    def show_triangles(self, img, landmark_points):
        try:
            size = img.shape
            rect = (0, 0, size[1], size[0])

            for coords in landmark_points:

                # round values in range
                coords[coords[:,0] >= size[1]] = size[1]-1
                coords[coords[:,1] >= size[0]] = size[0]-1
                coords[coords < 0] = 0

                coords = coords.astype(int).tolist()
                
                subdiv = cv2.Subdiv2D(rect)
                subdiv.insert(coords)
                
                img = self.draw_delaunay(img, subdiv)

        except Exception as e:
            print('Visualizer.show_triangles():', e)

        return img
            

    # Draw delaunay triangles
    def draw_delaunay(self, img, subdiv, delaunay_color=(255, 255, 255)) :

        triangle_list = subdiv.getTriangleList();
        size = img.shape
        r = (0, 0, size[1], size[0])
        alpha = 0.15
        
        overlay = img.copy()
        
        for t in triangle_list:
            
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])
            
            if  self.rect_contains(r, pt1) and \
                self.rect_contains(r, pt2) and \
                self.rect_contains(r, pt3):
            
                cv2.line(overlay, pt1, pt2, delaunay_color, 1, lineType=cv2.LINE_AA)
                cv2.line(overlay, pt2, pt3, delaunay_color, 1, lineType=cv2.LINE_AA)
                cv2.line(overlay, pt3, pt1, delaunay_color, 1, lineType=cv2.LINE_AA)
                
                img_new = cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)
                
        return img_new

    # Check if a point is inside a rectangle
    def rect_contains(self, rect, point):
        if point[0] < rect[0]:   return False
        elif point[1] < rect[1]: return False
        elif point[0] > rect[2]: return False
        elif point[1] > rect[3]: return False
        return True
