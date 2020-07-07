import os
import cv2
import numpy as np

class detect2label():
    def __init__(self):
        self.PWD_PATH = os.getcwd() #path cerrent
        self.WEIGHT_FILE = 'models_370000.weights'
        self.CFG_FILE = 'cfg_file.cfg'
        # self.IMAGE_NAME = 'origin_img.png'
        self.CONF_THRESH = 0.5 
        self.NMS_THRESH = 0.5 
        self.WEIGHT_PATH = os.path.join(self.PWD_PATH,'weight',self.WEIGHT_FILE) 
        self.CFG_PATH = os.path.join(self.PWD_PATH,'cfg',self.CFG_FILE)
        self.list_size_image = []
        self.type_file = ['jpg','png']
        self.list_file = []
        self.list_size_yolov3 = []
        self.list_filename = []

    def check_type_img(self,img_file):
        if self.split_typefile(img_file,type_file=True) in self.type_file:
            return True
        else:
            return False

    def list_dir(self,path_directory):
        for file_name in os.listdir(path_directory):
            check_type_imgs = self.check_type_img(file_name)
            if check_type_imgs:
                path_filename = os.path.join(path_directory,file_name)
                self.list_file.append(path_filename)
        return self.list_file

    def split_typefile(self,img_file,type_file = False,filename = False):
        file_split = img_file.split('.')
        if filename and type_file:
            return file_split[0],file_split[1]
        if filename:
            return file_split[0]
        if type_file:
            return file_split[1]
            

    def convert_xml_value2yolov3(self,list_size,width,height):
        # print(list_size[0][1])
        xmin = list_size[0][0]
        xmax = list_size[0][1]
        ymin = list_size[0][2]
        ymax = list_size[0][3]

        xcen = float(np.divide(np.divide((np.add(xmin,xmax)),2),widht))
        ycen = float(np.divide(np.divide((np.add(ymin,ymax)),2),height))

        w = float(np.divide(np.subtract(xmax,xmin),widht))
        h = float(np.divide(np.subtract(ymax,ymin),height))
        xcen = "%.6f" % xcen
        ycen = "%.6f" % ycen
        w = "%.6f" % w
        h = "%.6f" % h
        return xcen, ycen, w, h
    
    def echo_w_h(self,img):
        image = cv2.imread(img)
        w,h = image.shape[:2]
        return w,h,image

    def detection_by_yolov3(self,img,height,width):
        # Load the network
        net = cv2.dnn.readNetFromDarknet(self.CFG_PATH, self.WEIGHT_PATH)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # Get the output layer from YOLO
        layers = net.getLayerNames()
        output_layers = [layers[np.subtract(i[0],1)] for i in net.getUnconnectedOutLayers()]

        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), swapRB=True, crop=False)
        # blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_outputs = net.forward(output_layers)

        class_ids, confidences, b_boxes = [], [], []
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > self.CONF_THRESH:
                    center_x, center_y, w, h = (np.multiply(detection[0:4],np.array([width, height, width, height]))).astype('int')

                    x = int(np.subtract(center_x,np.divide(w,2)))
                    y = int(np.subtract(center_y,np.divide(h,2)))

                    b_boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(int(class_id))

        # Perform non maximum suppression for the bounding boxes to filter overlapping and low confident bounding boxes
        indices = cv2.dnn.NMSBoxes(b_boxes, confidences, self.CONF_THRESH, self.NMS_THRESH)
        self.list_size_yolov3.clear()
        for index in indices:
            # print(index.astype(int))
            out_images = np.array(b_boxes)[index.astype(int)]
            x1, y1, w, h = out_images[0]
            x2 = np.add(x1,w)
            y2 = np.add(y1,h)
            self.list_size_image.append([x1,x2,y1,y2])
            xcen, ycen, w, h = self.convert_xml_value2yolov3(self.list_size_image,width,height)
            self.list_size_yolov3.append([xcen, ycen, w, h])
            self.list_size_image.clear()
        return self.list_size_yolov3

    def list_name_from_directory(self,directory_path):
        for _filename in os.listdir(directory_path):
            if self.check_type_img(_filename):
                name_split = self.split_typefile(_filename,filename = True)
                self.list_filename.append(name_split)
        return self.list_filename

if __name__ == "__main__":
    path_directory = os.path.join(os.getcwd(),'img')
    tree = detect2label()
    list_pathfile = tree.list_dir(path_directory)
    list_filename = tree.list_name_from_directory(path_directory)
    for _file,save_filename in zip(list_pathfile,list_filename):
        widht , height , img = tree.echo_w_h(_file)
        list_size = tree.detection_by_yolov3(img,height,widht)
        file_saved = os.path.join(os.getcwd(),'img','{}.txt'.format(save_filename))
        with open(file_saved,'w') as txt_file:
            # if text not None:
            for text in list_size:
                txt_file.write('0 ')
                for separa_list in text:
                    txt_file.write('{} '.format(separa_list))
                txt_file.write('\n')
            # else:
            #     pass
