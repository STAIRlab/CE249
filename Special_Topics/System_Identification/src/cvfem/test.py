import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

class CorrodedCoordinateGeneration:
    def __init__(self, raw_img_path, mask_img_path,num_col,num_row):
        self.raw_img_path = raw_img_path
        self.mask_img_path = mask_img_path
        self.num_col = num_col
        self.num_row = num_row
        self.points = []
        self.raw_img = None
        self.mask_img = None
        self.selected_area = None

    def load_img(self):
        self.raw_img = cv2.imread(self.raw_img_path)
        self.mask_img = cv2.imread(self.mask_img_path, cv2.IMREAD_GRAYSCALE)
        if self.raw_img is None or self.mask_img is None:
            raise FileNotFoundError("One or both of the images could not be loaded.")
        self.yr_max = self.raw_img.shape[0]; self.xr_max = self.raw_img.shape[1]
        self.y_max = self.mask_img.shape[0]; self.x_max = self.mask_img.shape[1]

    def select_img_4points(self):
        '''
        Manually click four corners of the Sign Structure in order to calculate
        the area of the Sign
        '''
        self.raw_img = cv2.cvtColor(self.raw_img, cv2.COLOR_BGR2RGB)  # cv2 uses BGR while matplotlib uses RGB
        fig, ax = plt.subplots()
        ax.imshow(self.raw_img)
        ax.set_title('Click to select 4 points')

        def onclick(event):
            if len(self.points) < 4:
                x, y = event.xdata, event.ydata
                self.points.append((x, y))
                ax.plot(x, y, 'ro')
                plt.draw()
                if len(self.points) == 4:
                    fig.canvas.mpl_disconnect(cid)
                    self.points = self.order_points(self.points)
                    area = self.calculate_polygon_area(self.points)
                    self.selected_area = area

        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

    def order_points(self, points):
        '''
        This function ensures the order of points is listed in Clockwise (CW) direction.
        '''
        points = np.array(points)
        rect = np.zeros((4, 2), dtype='float32')
        s = points.sum(axis=1)
        rect[0] = points[np.argmin(s)]
        rect[2] = points[np.argmax(s)]
        diff = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diff)]
        rect[3] = points[np.argmax(diff)]
        return rect

    def calculate_polygon_area(self, points):
        '''
        Calculate the Area of Polygon formed by the four nodes selected before.
        Using Shoelace Formula or Gauss's area formula.
        Area = 1/2*(Σ(x_i*y_(i+1) - y_i*x_(i+1)) + x_n*y_1 - y_n*x_1)
        '''
        n = len(points)
        if n < 3:
            print("Error: At least 4 points should be clicked")
            return None
        area = 0
        for i in range(n):
            j = (i + 1) % n  # j = 2,3,4,1 while i = 1,2,3,4
            area += points[i][0] * points[j][1] - points[j][0] * points[i][1]
        area = abs(area) / 2.0
        return area
    
    def calculate_area_ratio_of_polygon_to_image(self):
        '''
        Calculate the ratio between the polygon area (The Sign Area) to the total image area
        '''
        if self.raw_img is not None and self.selected_area is not None:
            height, width, _ = self.raw_img.shape
            area_of_IMG = height * width
            area_of_sign = self.selected_area
            ratio_sign = area_of_sign / area_of_IMG
            print(f'Ratio Sign {ratio_sign}')
            return ratio_sign
        else:
            print("Error: Raw image or selected area not loaded")
            return None

    def calculate_area_ratio_of_corroded_to_mask_img(self):
        '''
        Calculate the ratio between the corroded area (white pixels) to the total mask image area
        '''
        if self.mask_img is not None:
            total_pixels = self.mask_img.shape[0] * self.mask_img.shape[1]
            white_pixel_count = np.sum(self.mask_img == 255)
            ratio_corroded = white_pixel_count / total_pixels
            print(f'Ratio Corroded {ratio_corroded}')
            return ratio_corroded
        else:
            print("Error: Mask image not loaded")
            return None
        
    def calculate_corroded_ratio(self):
        '''
        Calculate the ratio of corroded area to the sign area.
        The ratio is used for generation of FE model later.
        '''
        sign_area_ratio = self.calculate_area_ratio_of_polygon_to_image()
        corroded_area_ratio = self.calculate_area_ratio_of_corroded_to_mask_img()
        if sign_area_ratio is not None and corroded_area_ratio is not None:
            corroded_to_sign_ratio = corroded_area_ratio / sign_area_ratio
            print(f"Ratio of corroded area to sign area: {corroded_to_sign_ratio}")
        else:
            print("Error: Could not calculate the corroded to sign area ratio due to missing data.")
        self.corroded_to_sign_ratio = corroded_to_sign_ratio
    
    def interpolate_points(self,p1,p2,num_points):
        '''
        Interpolate points between p1 and p2.
        
        Parameters:
        p1, p2 (array-like): Endpoints between which interpolation is to be done.
        num_points (int): Number of points to interpolate.
        
        Returns:
        np.ndarray: Interpolated points.
        '''
        return np.linspace(p1,p2,num_points)
    
    def generate_grid(self,points):
        '''
        Generate the location of grid points in Mask Images.
        
        Parameters:
        points (array-like): Four corner points of the selected area.
        
        Returns:
        np.ndarray: Grid points mapped to the mask image.
        '''
        top_edge = self.interpolate_points(points[0],points[1],self.num_col+1)
        bottom_edge = self.interpolate_points(points[3],points[2],self.num_col+1)
        grid_points = []
        for i in range(self.num_col + 1):
            left_col = self.interpolate_points(top_edge[i], bottom_edge[i], self.num_row + 1)
            grid_points.append(left_col)
        self.grid_points = np.array(grid_points) # Shape (31,11,2)

        # Map the grid from raw img to mask img
        self.points_mask = self.points.copy()
        self.points_mask[:,0] = self.points[:,0] * self.x_max / self.xr_max
        self.points_mask[:,1] = self.points[:,1] * self.y_max / self.yr_max

        self.grid_points_mask = self.grid_points.copy()
        self.grid_points_mask[:, :, 1] = self.grid_points[:, :, 1] * self.y_max / self.yr_max
        self.grid_points_mask[:, :, 0] = self.grid_points[:, :, 0] * self.x_max / self.xr_max
        return self.grid_points_mask
    
    def draw_grid_fig(self):
        '''
        Draw a figure with the grid overlaid on the mask image.
        '''
        mask_img_color = cv2.cvtColor(self.mask_img, cv2.COLOR_GRAY2BGR)
        polygon = plt.Polygon(self.points_mask, closed=True, fill=None, edgecolor='r')
        plt.gca().add_patch(polygon)

        # Draw the grid points on the mask image
        for i in range(self.num_col + 1):
            for j in range(self.num_row + 1):
                cv2.circle(mask_img_color, (int(self.grid_points_mask[i, j, 0]), 
                                            int(self.grid_points_mask[i, j, 1])), 
                                            radius=3, color=(0, 0, 255), thickness=-1)
        plt.imshow(cv2.cvtColor(mask_img_color, cv2.COLOR_BGR2RGB))
        plt.xlim(0, self.x_max)
        plt.ylim(self.y_max, 0)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    def generate_box(self):
        '''
        Generate bounding boxes for the grid points in the mask image.
        
        Returns:
        np.ndarray: An array of bounding boxes with each box defined by [y1, y2, x1, x2].
        '''
        x_locs,y_locs,_ = self.grid_points_mask.shape
        self.boxes = np.array([])
        for yi in range(y_locs-1):
            for xi in range(x_locs-1):
                point1 = self.grid_points_mask[xi,yi,:]
                point2 = self.grid_points_mask[xi+1,yi+1,:]
                box = np.array([point1[1],point2[1],point1[0],point2[0]])
                if yi == 0 and xi ==0:
                    self.boxes = box
                else:
                    self.boxes = np.vstack((self.boxes, box))
        return self.boxes

    def detect_corrosion(self):
        '''
        Iterate every box in mask figure and detect the corrosion ratio.
        If the ratio is larger than 44%, save the corresponding coordinates of points
        into a cvs file for the Abaqus to excute.
        '''
        self.corroded_box = np.array([0,0,0,0])
        self.corroded_ratio = []
        self.corroded_point = np.array([0,0])
        self.corroded_point2 = np.array([0,0])
        mask_img_color = cv2.cvtColor(self.mask_img, cv2.COLOR_GRAY2BGR)
        
        for num_box in range(self.boxes.shape[0]):
            box = self.boxes[num_box,:].astype(int)
            test_box = self.mask_img[box[0]:box[1],box[2]:box[3]]
            # [y_loc:y_loc + box_height, x_loc:x_loc + box_width]
            white_pixel_count = np.sum(test_box == 255)
            x,y = np.shape(test_box)
            total_pixel_count = x * y
            ratio = white_pixel_count/total_pixel_count

            if ratio >= 0.5:
                self.corroded_point = np.vstack((self.corroded_point,np.array([box[2],box[0]])))
                self.corroded_point2 = np.vstack((self.corroded_point2,np.array([box[3],box[1]])))
                cv2.circle(mask_img_color, (box[2], box[0]), radius=3, color=(0, 0, 255), thickness=-1)
                cv2.circle(mask_img_color, (box[3], box[1]), radius=3, color=(0, 255, 255), thickness=-1)
            self.corroded_ratio.append(ratio)

        self.corroded_point = self.corroded_point[1:,:]
        self.corroded_point2 = self.corroded_point2[1:,:]
        df_point1 = pd.DataFrame(self.corroded_point, columns=['x1', 'y1'])
        df_point2 = pd.DataFrame(self.corroded_point2, columns=['x2', 'y2'])
        self.combined_df = pd.concat([df_point1, df_point2], axis=1)

        self.combined_df.to_csv("corroded_coordinates_mask.csv", index=False, header=False)

        plt.imshow(cv2.cvtColor(mask_img_color, cv2.COLOR_BGR2RGB))
        plt.xlim(0, self.x_max)
        plt.ylim(self.y_max, 0)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    def abaqus_coord_transformation(self):
        '''
        Transform corroded points coordinates to Abaqus coordinate system.
        '''
        Abaqus_4pointCoord = np.array([
            [22.25  , 276   ],
            [205.125, 276   ],
            [205.125, 213.75],
            [22.25  , 213.75]
        ], dtype=np.float32)
        point1_abaqus_list = np.array([0,0])
        point2_abaqus_list = np.array([0,0])
        a = np.array(self.points_mask,dtype = np.float32)
        M = cv2.getPerspectiveTransform(a, Abaqus_4pointCoord)
        
        for i in range(self.corroded_point.shape[0]):
            point1 = np.array(self.corroded_point[i],dtype = np.float32)
            point1_abaqus = cv2.perspectiveTransform(np.array([[point1]]), M).reshape(-1)
            point1_abaqus_list=np.vstack((point1_abaqus_list,point1_abaqus))

            point2 = np.array(self.corroded_point2[i],dtype = np.float32)
            point2_abaqus = cv2.perspectiveTransform(np.array([[point2]]), M).reshape(-1)
            point2_abaqus_list=np.vstack((point2_abaqus_list,point2_abaqus))

        point1_abaqus_list = point1_abaqus_list[1:,:]
        point2_abaqus_list = point2_abaqus_list[1:,:]

        df_point1_abaqus = pd.DataFrame(point1_abaqus_list)
        df_point2_abaqus = pd.DataFrame(point2_abaqus_list)
        self.combined_df_abaqus = pd.concat([df_point1_abaqus, df_point2_abaqus], axis=1)
        self.combined_df_abaqus.to_csv("corroded_coordinates_abaqus.csv", index=False, header=False)

    def process(self):
        self.load_img()
        self.select_img_4points() 
        self.calculate_corroded_ratio()
        self.generate_grid(self.points)
        self.draw_grid_fig()
        self.generate_box()
        self.detect_corrosion()
        # self.abaqus_coord_transformation()
        
if __name__ == "__main__":
    raw_img_path = 'C:/Users/19461/Desktop/CE299/Vision_based_area_detection/raw/raw.JPG'
    mask_img_path = 'C:/Users/19461/Desktop/CE299/Vision_based_area_detection/mask/0_mask.png'

    area_detection = CorrodedCoordinateGeneration(raw_img_path, mask_img_path,num_col=30,num_row=10)
    area_detection.process()