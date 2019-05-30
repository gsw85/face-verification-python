
import copy
import cv2

class FaceVisualizer:

    FONT_TYPE      = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE     = 0.5
    FONT_THICKNESS = 1
    FONT_COLOR     = (20, 20, 20) # BGR
    BOX_COLOR      = (255, 255, 255) # BGR
    BOX_THICKNESS  = 1

    def show_boxes(self, img, bounding_boxes):
        for box in bounding_boxes:
            cv2.rectangle(
                img,
                (box['xmin'], box['ymin']),
                (box['xmax'], box['ymax']),
                self.BOX_COLOR,
                self.BOX_THICKNESS
            )

    def show_landmarks(self, img, landmark_points, recognized_faces, triangles=True):
        try:
            size = img.shape
            rect = (0, 0, size[1], size[0])
            
            # loop over facial landmarks
            for ind, coords in enumerate(landmark_points):
                # loop over the (x, y)-coordinates for the facial landmarks
                for (x, y) in coords:
                    cv2.circle(img, (x, y), 1, self.BOX_COLOR, thickness=-1)

                img = self.show_text_label(img, recognized_faces, ind, coords)

                if triangles:
                    img = self.draw_delaunay(img, coords, size, rect)

        except Exception as e:
            print('Visualizer.show_landmarks():', e)

        return img

    def show_text_label(self, img, recognized_faces, ind, coords):
        # get person name
        text = recognized_faces[ind]

        # location of the right eye brow
        x, y = coords[25]
        xmin, ymin = (x-10, y-10)

        # get text size
        size = cv2.getTextSize(text, self.FONT_TYPE, self.FONT_SCALE, thickness=2)
        text_width = size[0][0]
        text_height = size[0][1]

        padding = 1
        cv2.rectangle(
            img,
            # bottom-left
            (xmin-padding, ymin+padding),
            # top-right
            (xmin+text_width+padding, ymin-text_height-padding), 
            self.BOX_COLOR, thickness=-1
        )

        cv2.putText(
            img, recognized_faces[ind], (xmin, ymin), 
            self.FONT_TYPE, self.FONT_SCALE, self.FONT_COLOR, self.FONT_THICKNESS
        )

        return img


    # Draw delaunay triangles
    def draw_delaunay(self, img, coords, size, rect, delaunay_color=(255, 255, 255)):

        # round values in range-5
        # e.g. width = 100, then round to 95
        coords[coords[:,0] >= size[1]] = size[1]-5
        coords[coords[:,1] >= size[0]] = size[0]-5
        coords[coords < 0] = 5

        coords = coords.astype(int).tolist()
        
        # create Subdiv2D
        subdiv = cv2.Subdiv2D(rect)
        subdiv.insert(coords)

        # get triangle list
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
