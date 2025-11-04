import glob
import math
import numpy as np
import tqdm
import os
import pandas as pd
import shutil
import re
import warnings
import datetime
import torch

from PIL.ExifTags import TAGS
from time import sleep
from PIL import Image

class DLR_Detection(object):

    def __init__(self,R,reference,Trees,livedet,generaldet,camera_weight,crop,show_label):
        self.R = R
        self.reference = reference
        if self.reference[0] == 0:
            self.reference = None
        
        self.Trees = Trees - 1
        self.livedet  = livedet
        self.generaldet = generaldet
        self.split = False
        self.combined = False
        self.camera_weight = camera_weight

        self.class_names = ['B', 'K', 'C', 'A']
        self.class_numbers = 4
        self.classes = list(range(self.class_numbers))

        self.crop = crop   
        
        self.show_label = show_label

    def loadimages(self,images_path):
        """
        Load images from the designated folder and setup several variables and flags \n
        Also extracts the GPS coordinates from the images \n
        """
        # Open an image to check if they are already combined

        try:
            img = Image.open(glob.glob(images_path)[0])
            if img.height/img.width > 3 or img.width/img.height > 3:
                self.combined = True
            else:
                self.combined = False
        except:
            pass
        # Image loading is not necessary for general or live detection
        if self.livedet or self.generaldet:
            self.images_path = images_path
            self.src_folder = os.path.split(images_path)[0]

        else:
            src_folder = os.path.split(images_path)[0]
            self.src_folder = src_folder

            # Check if images have already been split into three camera folders
            if os.path.isdir(os.path.join(src_folder,'camera1'))\
                and os.path.isdir(os.path.join(src_folder,'camera2'))\
                and os.path.isdir(os.path.join(src_folder,'camera3'))\
                and not self.combined\
                and not len(glob.glob(os.path.join(src_folder,'camera1','*.jpeg'))) == 0:
                self.split = True
                camera1path = os.path.join(src_folder,'camera1','*.jpeg')
                camera2path = os.path.join(src_folder,'camera2','*.jpeg')
                camera3path = os.path.join(src_folder,'camera3','*.jpeg')
                camerapaths = [camera1path,camera2path,camera3path]        
                images_path = camera1path

            filenames = []
            Coords = []
            imgsz = []

            # Load all images from target folder
            print(f'Loading images from {images_path}')
            pbar = tqdm.tqdm(total=len(glob.glob(images_path)), bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
            for filename in glob.glob(images_path):
                image=Image.open(filename)
                if image.width > image.height: # Rotate image if necessary
                    exif = image.getexif()
                    image.rotate(angle=270,expand=True).save(filename,exif = exif)
                    
                exif = {}

                for tag, value in image._getexif().items():
                    if tag in TAGS:
                        exif[TAGS[tag]] = value
                # Get GPS Coordinates
                GPSInfo = exif["GPSInfo"]
                GPS_Coords = [[GPSInfo[2]],[GPSInfo[4]],GPSInfo[6]]
                temp = GPS_Coords[0]
                Lat = float(temp[0][0]) + float (temp[0][1])/60 + float(temp[0][2])/3600
                temp = GPS_Coords[1]
                Lon = float(temp[0][0]) + float(temp[0][1])/60 + float(temp[0][2])/3600
                H = float(GPS_Coords[2])
                
                # Convert Lat and Lon to cartesian coordinates
                x = self.R * math.cos(np.deg2rad(Lat)) * math.cos(np.deg2rad(Lon))
                y = self.R * math.cos(np.deg2rad(Lat)) * math.sin(np.deg2rad(Lon))
                Coords.append(np.array((x, y)))
                
                # Get filename and image sizes
                imgsz.append(image.size)
                filenames.append(filename)


                pbar.update(1)

            if self.split: # Rotate the other camera images if necessary
                for images_path in camerapaths[1:3]:
                    for filename in glob.glob(images_path):
                        image = Image.open(filename)
                        if image.width > image.height:
                            exif = image.getexif()
                            image.rotate(angle=270,expand=True).save(filename,exif = exif)
                        exif = {}
                self.cameras = [filenames,glob.glob(camera2path),glob.glob(camera3path)]

            self.imgsz = imgsz # Image size of the loaded images
            self.filenames = filenames # Filenames of all loaded images
            self.coords = Coords # Coordinates extracted from the loaded images

            pbar.close()

            print(f'Finished loading {len(glob.glob(images_path))} images')
    
    def filterCoords(self):
        """
        Set up filter coordinates \n
        This creates a list of coordinates based on the first and last tree in a row and the number of trees in a row \n
        Assumes equal distance between every tree \n
        """

        FilteredCoords = []

        # Set reference coordinates
        if self.reference is not None: # Extract reference form given reference coordinates
            coords = self.reference    
            x1 = float(coords[0])      
            y1 = float(coords[1])
            x2 = float(coords[2])
            y2 = float(coords[3])

            # Convert image coordinates to latittude and longitude if necessary
            if x1 < 1000:
                x1cat = self.R * math.cos(np.deg2rad(x1)) * math.cos(np.deg2rad(y1))
                y1cat = self.R * math.cos(np.deg2rad(x1)) * math.sin(np.deg2rad(y1)) 
                x2cat = self.R * math.cos(np.deg2rad(x2)) * math.cos(np.deg2rad(y2))
                y2cat = self.R * math.cos(np.deg2rad(x2)) * math.sin(np.deg2rad(y2))

            first =np.array((x1cat, y1cat))
            last = np.array((x2cat, y2cat))
        else: 
            warnings.warn("No reference coordinates have been set, coordinates will be extracted from images, please set reference coordinates maually for more accurate filtering", stacklevel=2)
            coords = self.coords # Extract reference coordinates from images
            first = coords[0]    # Image coordinates are not accurate
            last = coords[-1]    # Please set the reference maunally if you want more accurate results

        # Set stepsize
        diff = np.subtract(last,first)
        Stepsize = diff/self.Trees
        Step = first
        FilteredCoords.append(Step)

        # Build filtered coordinates
        while (round(Step[0],7) != round(last[0],7)) and (round(Step[1],7) != round(last[1],7)):
            Step = Step + Stepsize
            FilteredCoords.append(Step)
        self.FilteredCoords = FilteredCoords

        # Set distance between two trees
        self.tree_distance = np.linalg.norm(FilteredCoords[0]-FilteredCoords[1])

    def BasicNoFilter(self,model,multirowaverage=False):
        """
        Basic method of detection \n
        Images are assumed to be pre-filtered -> one tree per image \n
        Handles other cases (multiple cameras, combined images) automatically \n
        """

        images = []
        cameras = [[],[],[]]

        # Check if images are split into three camera folders
        dec  = os.path.isdir(os.path.join(self.src_folder,'camera1'))\
        and os.path.isdir(os.path.join(self.src_folder,'camera2'))\
        and os.path.isdir(os.path.join(self.src_folder,'camera3')) \
        and not self.combined\
        and not len(glob.glob(os.path.join(self.src_folder,'camera1','*.jpeg'))) == 0

        # Case: multiple images
        if self.split or dec:
            self.split = True
            camera1path = os.path.join(self.src_folder,'camera1','*.jpeg')
            camera2path = os.path.join(self.src_folder,'camera2','*.jpeg')
            camera3path = os.path.join(self.src_folder,'camera3','*.jpeg')
            camerapaths = [camera1path,camera2path,camera3path]

            # Run detection three times for each camera
            results = []
            for i in range(3):
                results.append(model(glob.glob(camerapaths[i])))
            if not multirowaverage:
                self.saveresults(results=results,general=True,multicamera=True)

        # Case: single image
        else:
            for filename in glob.glob(self.images_path):
                
                # Case: three images are combined into a single image
                if self.combined:
                    im = Image.open(filename)
                    
                    h = im.height
                    hs = h/3

                    if self.crop:
                        im3 = im.crop((0,0,im.width,hs*(3/4)))
                    else:
                        im3 = im.crop((0,0,im.width,hs))
                    cameras[0].append(im3)
                    im2 = im.crop((0,hs,im.width,hs*2))
                    cameras[1].append(im2)
                    im1 = im.crop((0,hs*2,im.width,h))
                    cameras[2].append(im1)
                images.append(filename)
            

            if self.combined: 
                results = []
                for i in range(3):
                    results.append(model(cameras[i]))
                if not multirowaverage:
                    self.saveresults(results=results,general=True,multicamera=True)

            # Case: images are not combined
            else:
                images.sort()
                results = model(images)
                if not multirowaverage:
                    self.saveresults(results=results,general=True)
        return results       

    def BasicAutoFilter(self,model): 
        """
        Basic method of detection \n
        Images will be automatically filtered according to the set reference coordinates \n
        """

        # Go through all filtered coordinates
        detections = np.zeros((1,len(self.classes)))
        total = 0
        results_all_save = []
        pbar = tqdm.tqdm(total=len(self.FilteredCoords), bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        for i in self.FilteredCoords:
            t = []
            # Go through all image coordinates
            for Coord in self.coords:
                # Calculate the distance of the image to every filtered coordinate
                d = np.linalg.norm(Coord-i)
                t.append(d)
            # Find the closest distance and get the index of that image
            index = np.argmin(t)

            # Run Detect in the filtered image folder
            results_all = model(self.filenames[index])
            results = results_all.pandas().xyxy[0]

            # Get number of classes from results
            temp = np.zeros((1,self.class_numbers))
            for c in self.classes:
                temp[0][c] = len(results[results['class'] == c])
            detections = np.append(detections,temp,0)

            # Compute the area of the classes
            results = np.array(results)
            if results.size == 0:
                pbar.update()
                continue

            results_all_save.append(results_all)
            pbar.update(1)
        pbar.close()    

        detections = np.round(detections)
        detections = detections.squeeze()
        det = pd.DataFrame(detections,columns = self.class_names)
        det = det.drop(labels = 0, axis = 0)
        return results_all_save,det
    
    def WmeanAutoFilter(self, model):
        """
        Apply a weighted mean between neighbouring trees when detecting apples \n
        Images will be automatically filtered according to the set reference coordinates \n
        """
        total_weight = 0
        batch = []
        detections = []
        results_all_save = []
        counter = 1

        pbar = tqdm.tqdm(total=len(self.FilteredCoords), bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        count = 0
        for i in self.FilteredCoords:
            count = count +1
            t = []

            # Go through all image coordinates
            for Coord in self.coords:
                # Calculate the distance of every image to the filtered coordinate
                d = np.linalg.norm(Coord-i)
                t.append(d)

            # Find index of all images within a certain threshold around the filtered coordinate
            index = [j for j in range(len(t)) if t[j] < self.tree_distance/2]
            tree_index = np.argmin(t) # index of the tree closest to the coordinate
            if not len(index):
                continue
            else:
                # Run Detect in the filtered image folder
                for idx in index:
                    results_all = model(self.filenames[idx])
                    if idx == tree_index:
                        results_all_save.append(results_all)
                    results = results_all.pandas().xyxy[0]

                    single = np.zeros((1,len(self.classes)))
                    for j in range(len(self.classes)):
                        single[0][j] = len(results[results['name']==self.class_names[j]])
                    weight =1 - (t[idx]/(self.tree_distance/2))
                    total_weight = total_weight + weight
                    batch.append(single*weight)

                # Saving model results
                try:
                    detections.append(sum(batch)/total_weight)
                except:
                    print('Total weight was zero')

                pbar.update(1)

                # Starting new iteration
                counter = counter + 1
                batch = []
                total_weight = 0
        
        pbar.close()
        
        # Saving results
        detections = np.round(detections)
        detections = detections.squeeze()
        det = pd.DataFrame(detections,columns = self.class_names)
        det = det.drop(labels = 0, axis = 0)
        return results_all_save, det
    
    def live(self,model):
        """
        This method simply checks the designaded folder continuously and feeds all images in the folder to the model \n
        """

        print(self.classes)
        print('Starting live detection')
        print('Checking for images in ' +  str(self.images_path))
        total = 0
        stop = 0
        imgsz = []
        detections = []
        # Start loop
        while True:
            # Get all files in directory 
            filename =  glob.glob(self.images_path)

            # Check if there are files in the directory
            if len(filename) == 0:
                print('No file in directory waiting 3 seconds and checking folder again')
                sleep(3)
                stop += 1

                # End programm if no files have been added
                if stop == 2:
                    print('Closing programm')
                    try:
                        detections = np.array(detections).squeeze()
                        det = pd.DataFrame(detections,columns = self.classes)
                        det.to_csv(path + '/results.csv', index=False)
                        return
                    except:
                        return
            else:

                path,file_=os.path.split(filename[0])

                DLR_Detection.make_directories(self,['seen','results'],path)

                stop = 0
                print('Found ' + str(len(filename)) + ' images')
                print('Starting detection')
                # Iterate over all images in the directory
                for filename in glob.glob(self.images_path):
                    
                    path,file_=os.path.split(filename)

                    # Open the image
                    image=Image.open(filename)
                    imgsz.append(image.size)
                    
                    # Feed image to the network
                    results_all = model(filename)
                    results_all.save(save_dir=path + '/results')
                    results = results_all.pandas().xyxy[0]
                    
                    temp = np.zeros((1,len(self.classes)))
                    for i in range(len(self.classes)):
                        temp[0][i] = len(results[results['class']==self.classes[i]])
                    detections.append(temp)
                    
                    
                    # Compute the area of the classes
                    results = np.array(results)
                    #area,total = self.computearea(results,imgsz,total)
                    total = 0
                        
                    # Move image to new directory
                    image.close()
                    shutil.move(filename, path + '/seen/' + file_)
                    
    def getversion(self,version):
        """
        Sets the class names and number of classes based on the version \n
        DEPRECATED \n
        """
        if version == 1:
            self.class_names = ['B', 'K']
            self.class_numbers  = 2
            self.classes = list(range(self.class_numbers))
        elif version == 2:
            self.class_names = ['B', 'K', 'C']
            self.class_numbers = 3
            self.classes = list(range(self.class_numbers))
        else:
            self.class_names = ['B', 'K', 'C', 'A']
            self.class_numbers = 4
            self.classes = list(range(self.class_numbers))    

    def make_directories(self,names,path):

        dir_paths = []
        for name in names:
            dir = name
            dir_path = os.path.join(path, dir)
            dir_paths.append(dir_path)
            self.dir_paths = dir_paths
            try:
                shutil.rmtree(dir_path)
            except:
                pass
            try:
                os.mkdir(dir_path)
            except:
                pass
        return(dir_paths)

    def computearea(self,results,total, index = None):
        """
        This method computes the area in the image that is covered by the flower class \n
        UNUSED \n
        """
        for j in range(len(results)):
            if 'B' in results:
                if results[j][6] != 'B':
                    class_area = (results[j][2]-results[j][0])*(results[j][3]-results[j][1])
                    total = total + class_area
            else: 
                class_area = (results[j][2]-results[j][0])*(results[j][3]-results[j][1])
                total = total + class_area

            area = (total/(self.imgsz[index][0]*self.imgsz[index][1]))*100
        return(area,total)
    
    def splitcameras(self,images_path):
        """
        Splits all images in a folder into three camera subfolders \n
        Folder asssignment is based on filename \n
        Assumes that filename contains "camera0X" with X being the camera number \n
        Will also crop area from the top camera to remove overlap.\n
        Overlap assumes that camera03 is top and camera01 is bottom.
        """ 
        filenames = glob.glob(images_path)

        if filenames:
            camera1 = []
            camera2 = []
            camera3 = []
            cameras = pd.DataFrame()
            path,file_=os.path.split(images_path)

            DLR_Detection.make_directories(self,['camera1','camera2','camera3'],path)

            print(f"Separating images into camera folders")
            for filename in tqdm.tqdm(filenames):
                image=Image.open(filename)
                if image.width > image.height:
                    image=Image.open(filename)
                    exif = image.getexif()
                    image.rotate(angle=270,expand=True).save(filename,exif = exif)
                image.close()
                split = filename.split('\\')
                file = split[-1]

                if re.search("camera01",filename):
                    camera1.append(path + '/camera1/' + file)
                    shutil.move(filename, path + '/camera1/' + file)

                elif re.search("camera02",filename):
                    camera2.append(path + '/camera2/' + file)
                    shutil.move(filename, path + '/camera2/' + file)

                elif re.search("camera03",filename):
                    camera3.append(path + '/camera3/' + file)
                    if self.crop:
                        im = Image.open(filename)
                        im = im.crop((0,0,im.width,im.height*(3/4)))
                        im.save(path + '/camera3/' + file)
                        os.remove(filename)
                    else:
                        shutil.move(filename, path + '/camera3/' + file)
            try:
                cameras['camera1'] = camera1
                cameras['camera2'] = camera2
                cameras['camera3'] = camera3
            except:
                self.EqualizeFolders(path,camera1,camera2,camera3)
                cameras = pd.DataFrame()
                cameras['camera1'] = glob.glob(os.path.join(path,'camera1/*.jpeg'))
                cameras['camera2'] = glob.glob(os.path.join(path,'camera2/*.jpeg'))
                cameras['camera3'] = glob.glob(os.path.join(path,'camera3/*.jpeg'))
            self.cameras = cameras
            self.split = True

        else:
            images_path = os.path.split(images_path)[0]
            cameras = pd.DataFrame()
            cameras['camera1'] = glob.glob(os.path.join(images_path,'camera1/*.jpeg'))
            cameras['camera2'] = glob.glob(os.path.join(images_path,'camera2/*.jpeg'))
            cameras['camera3'] = glob.glob(os.path.join(images_path,'camera3/*.jpeg'))
            self.cameras = cameras
            self.split = True

    def combineimages(self, images_path):
        """
        Method for combining the three camera images into one \n
        Images will be stacked horizontally and saved as a new file \n
        Can be useful for maunal filtering of the images afterwards \n
        """

        path,file = os.path.split(images_path[0])

        print(f"Combining images")
        for i in tqdm.tqdm(range(len(self.cameras))):

            im1 = Image.open(self.cameras['camera1'][i])
            im2 = Image.open(self.cameras['camera2'][i])
            im3 = Image.open(self.cameras['camera3'][i])

            merged_im  = Image.new('RGB', (im1.width, im1.height*3))
            merged_im.paste(im3, (0, 0))
            merged_im.paste(im2, (0, im1.height))
            merged_im.paste(im1, (0, im1.height*2))

            exif = im1.getexif()
            filenumber = createfilenumber(len(self.cameras),i)
            merged_im.save(path+ '/' + filenumber+'.jpeg',exif = exif)
        
    def EqualizeFolders(self,images_path,camera1,camera2,camera3):
        """
        Method to equalize the camera folders after the images have been split \n
        All folders must contain the same number of images for the detection to work \n
        Removes the last image in the folder until all folders contain the same number of files \n
        """

        len_camera1 = len(glob.glob(os.path.join(images_path,'camera1/*.jpeg')))
        len_camera2 = len(glob.glob(os.path.join(images_path,'camera2/*.jpeg')))
        len_camera3 = len(glob.glob(os.path.join(images_path,'camera3/*.jpeg')))
        while len_camera1 != len_camera2 or len_camera1 != len_camera3 or len_camera2 != len_camera3:
            if len_camera1 > len_camera2:
                os.remove(camera1[-1])
                len_camera1 = len(glob.glob(os.path.join(images_path,'camera1/*.jpeg')))
                camera1 = glob.glob(os.path.join(images_path,'camera1/*.jpeg'))
            elif len_camera1 > len_camera3:
                os.remove(camera1[-1])
                len_camera1 = len(glob.glob(os.path.join(images_path,'camera1/*.jpeg')))
                camera1 = glob.glob(os.path.join(images_path,'camera1/*.jpeg'))
            elif len_camera2 > len_camera1:
                os.remove(camera2[-1])
                len_camera2 = len(glob.glob(os.path.join(images_path,'camera2/*.jpeg')))
                camera2 = glob.glob(os.path.join(images_path,'camera2/*.jpeg'))
            elif len_camera2 > len_camera3:
                os.remove(camera2[-1])
                len_camera2 = len(glob.glob(os.path.join(images_path,'camera2/*.jpeg')))
                camera2 = glob.glob(os.path.join(images_path,'camera2/*.jpeg'))
            elif len_camera3 > len_camera1:
                os.remove(camera3[-1])
                len_camera3 = len(glob.glob(os.path.join(images_path,'camera3/*.jpeg')))
                camera3 = glob.glob(os.path.join(images_path,'camera3/*.jpeg'))
            elif len_camera3 > len_camera2:
                os.remove(camera3[-1])
                len_camera3 = len(glob.glob(os.path.join(images_path,'camera3/*.jpeg')))
                camera3 = glob.glob(os.path.join(images_path,'camera3/*.jpeg'))

    def saveresults(self,results=None,detections=None,general=False,multicamera=False):
        
        print(f"Saving results")
        if general:
    
            date = datetime.datetime.now().strftime("%d-%m-%Y %H-%M-%S")
            dir_path = os.path.join(".","results",date)
            try:
                os.mkdir(dir_path)
            except:
                sleep(1)
                date = datetime.datetime.now().strftime("%d-%m-%Y %H-%M-%S")
                dir_path = os.path.join(".","results",date)
                os.mkdir(dir_path)

            if not multicamera:
                save_path_images= os.path.join(dir_path,"Images")
                results.save(save_dir = save_path_images,labels=self.show_label)
                l = 1
                totals = np.zeros((1,len(self.class_names)))
                results = [results]
            else:
                l = 3
                totals = np.zeros((3,len(self.class_names)))
                label = ["Camera1","Camera2","Camera3"]
                for i in range(3):
                    save_path_images=os.path.join(dir_path,"Images",f"camera{i+1}")
                    results[i].save(save_dir=save_path_images,labels=self.show_label)

            detections_list = []
            for n in range(l):
                detections = results[n].pandas().xyxy[:]
                detections_list_camera = []
                for det in detections:
                    class_totals = []
                    for class_,i in zip(self.class_names,range(len(self.class_names))):
                        class_total = len(det[det["name"]==class_])
                        class_totals.append(class_total)
                        if multicamera:
                            totals[n,i] = totals[n,i] + class_total
                        else:
                            totals[0,i] = totals[0,i] + class_total
                    detections_list_camera.append(class_totals)
                if n == 1:
                    detections_list_camera = [[self.camera_weight*t1 for t1 in sub] for t,sub in enumerate(detections_list_camera)]  
                detections_list.append(detections_list_camera)
                
            pd.DataFrame(data=np.around(np.array(np.sum(detections_list,axis=0))),columns=self.class_names).to_csv(os.path.join(dir_path,"detections_all.txt"),sep=' ')
            df = pd.DataFrame(data=totals.astype(int),columns=self.class_names)
            if not multicamera:
                df.to_csv(os.path.join(dir_path,"totals.txt"),sep=' ')
                print(f"Done!\n")
                print(f"Total amount of apples detected: {sum(df['A'])}")
            else:
                df.index = label
                apple_total = 0
                for i in range(len(df["A"])):
                    if i == 1:
                        apple_total = apple_total + df['A'][i]*self.camera_weight
                    else:
                        apple_total = apple_total + df['A'][i]
                print(f"Done!\n")
                print(f"Total amount of apples detected: {apple_total}")
                df.to_csv(os.path.join(dir_path,"totals.txt"),index=True,sep=' ')
            
        else:
            date = datetime.datetime.now().strftime("%d-%m-%Y %H-%M-%S")
            dir_path = os.path.join(".","results",date)
            try:
                os.mkdir(dir_path)
            except:
                sleep(1)
                date = datetime.datetime.now().strftime("%d-%m-%Y %H-%M-%S")
                dir_path = os.path.join(".","results",date)
                os.mkdir(dir_path)

            if not multicamera:
                save_path_images= os.path.join(dir_path,"Images")
                l = 1
                totals = np.zeros((1,len(self.class_names)))
                results = [results]

                for class_,i in zip(self.class_names,range(len(self.class_names))):
                    class_total = np.sum(detections[class_])
                    totals[0,i] = class_total
            else:
                l = 3
                totals = np.zeros((3,len(self.class_names)))
                label = ["Camera1","Camera2","Camera3"]

                for n in range(l):
                    for class_,i in zip(self.class_names,range(len(self.class_names))):
                        class_total = np.sum(detections[n][class_])
                        totals[n,i] = class_total
                
                detections = detections[0].add(detections[1].add(detections[2]),fill_value=0)
            
            for n in range(l):
                if multicamera:
                    save_path_images=os.path.join(dir_path,"Images",f"camera{n+1}")
                for result in results[n]:
                    result.save(save_dir=save_path_images,labels=self.show_label)
        
            detections.round(decimals=0).to_csv(os.path.join(dir_path,"detections_all.txt"),sep=' ')
            df = pd.DataFrame(data=totals.astype(int),columns=self.class_names)
            if not multicamera:
                df.to_csv(os.path.join(dir_path,"totals.txt"),sep=' ')
                print(f"Done!\n")
                print(f"Total amount of apples detected: {sum(df['A'])}")
            else:
                df.index = label
                apple_total = 0
                for i in range(len(df["A"])):
                    if i == 1:
                        apple_total = apple_total + df['A'][i]*self.camera_weight
                    else:
                        apple_total = apple_total + df['A'][i]
                print(f"Done!\n")
                print(f"Total amount of apples detected: {apple_total}")
                df.to_csv(os.path.join(dir_path,"totals.txt"),index=True,sep=' ')

#%%
def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)

def createfilenumber(l,number):
    l = len(str(l))
    number = str(number)
    for i in range(l):
        if len(number) < i+1:
            number = '0' + number
    return(number)

def DLRNotebook(general=None,combine=None,splitcameras=None,multirowaverage=None,trees=None,references=None,imgpaths=None,conf_thresh=None,version=None,weights=None,live=None,wmean=None,crop=None,show_label=None):
    
    camera_weight = 0.1
    row_weight = 0.2

    # Fix variables
    if type(trees) is not list:
        trees = [trees]

    if type(references[0]) is not list:
        references = [references]

    while len(trees) != len(imgpaths) and len(references) != len(imgpaths):
        trees.append(trees[0])
        references.append(references[0])

    # World radius
    R = 6371008.7714 

    # Load local model
    print(f"Loading model")
    current_directory = os.getcwd()
    yolo_path = str(os.getcwd() + '/yolov5')

    model = torch.hub.load(yolo_path, 'custom', path=str(current_directory + weights), source='local',verbose= False)  # local repo
    model.conf = conf_thresh # Set confidence threshold
    print(f"Done!\n")
    
    if multirowaverage:
        results_all_multirow,detections_multirow = [],[]

    for imgpath,trees,reference in zip(imgpaths,trees,references):

        DLR = DLR_Detection(R,reference=reference,Trees=trees,livedet=live,generaldet=general,camera_weight=camera_weight,crop=crop,show_label=show_label) # Initiate

        if splitcameras:
            DLR.splitcameras(imgpath) # Split all images into three seperate folders based on camera number -> will always split into three cameras!
            if combine:
                DLR.combineimages(imgpaths) # Useful for manual filtering

        # Load all images and extract Lat, Lon and Height
        DLR.loadimages(imgpath)

        if not live and not general:
            DLR.filterCoords()

        # Get specified version
        #DLR.getversion(version)

        # General detection no filter
        if general:
            print(f"Starting general Detection")
            results_all = DLR.BasicNoFilter(model,multirowaverage=multirowaverage)

        # Live detection
        elif live:
            DLR.live(model=model) 

        # Detection with auto filter
        else:

            # Auto filter with weighted mean appllied
            if wmean:

                # Multiple cameras
                if DLR.split:
                    print(f"Starting weighted mean detection with automatic filtering for three cameras")
                    results_all,detections = [],[]
                    for i in range(3):
                        DLR.filenames = DLR.cameras[i]
                        if i==1:
                            results,det = DLR.WmeanAutoFilter(model=model)
                            results_all.append(results)
                            detections.append(det*camera_weight)
                        else:
                            results,det = DLR.WmeanAutoFilter(model=model)
                            results_all.append(results)
                            detections.append(det)
                    if not multirowaverage:
                        DLR.saveresults(results=results_all,detections=detections,multicamera=True)

                # Single camera        
                else:
                    print(f"Starting weighted mean detection with automatic filtering")
                    results_all=DLR.WmeanAutoFilter(model=model)
                    if not multirowaverage:
                        DLR.saveresults(results=results_all)

            # Auto filter without weighted mean applied        
            else:

                # Multiple cameras
                if DLR.split:
                    print(f"Starting detection with automatic filtering") 
                    results_all,detections = [],[]
                    for i in range(3):
                        DLR.filenames = DLR.cameras[i]
                        if i==1:
                            results,det = DLR.BasicAutoFilter(model=model)
                            results_all.append(results)
                            detections.append(det*camera_weight)
                        else:
                            results,det = DLR.BasicAutoFilter(model=model)
                            results_all.append(results)
                            detections.append(det)
                    if not multirowaverage:
                        DLR.saveresults(results=results_all,detections=detections,multicamera=True)

                # Single camera
                else:   
                    print(f"Starting detection with automatic filtering for three cameras") 
                    results_all,detections=DLR.BasicAutoFilter(model=model)
                    if not multirowaverage:
                        DLR.saveresults(results=results_all,detections=detections)

        if multirowaverage:
            results_all_multirow.append(results_all)

    # Average between east and west row
    if multirowaverage:  
        date = datetime.datetime.now().strftime("%d-%m-%Y %H-%M-%S")
        dir_path = os.path.join(".","results",date)
        os.mkdir(dir_path)
        class_names = DLR.class_names

        totals_all = []

        for n in range(len(results_all_multirow[0])): 
            detections_east = results_all_multirow[0][n].pandas().xyxy[:]
            detections_west = results_all_multirow[1][n].pandas().xyxy[:]
            total = [[] for i in range(len(class_names))]
            
            for det_east,det_west in zip(detections_east,detections_west):
                class_totals_east = []
                class_totals_west = []
                det_west.iloc[::,-1] # Reverse order of west row to match indices of east row

                for class_,i in zip(class_names,range(len(class_names))):
                    class_totals_east = len(det_east[det_east["name"]==class_])
                    class_totals_west = len(det_west[det_west["name"]==class_])

                    # Check which side has more detections
                    # The side with less detections is weighted lower
                    if class_totals_east > class_totals_west:
                        total[i].append(class_totals_east +  class_totals_west*row_weight)
                    else:
                        total[i].append(class_totals_east*row_weight +  class_totals_west)

            df = pd.DataFrame(total).transpose()
            df.columns = class_names
            df = df.round()
            totals_all.append(df)

        if DLR.split or DLR.combined:
            l = 3
            label = ["Camera1","Camera2","Camera3"]
            camera_totals = np.zeros((l,len(class_names)))
            df_final = totals_all[0].add(totals_all[1]*camera_weight).add(totals_all[2])
            df_final = df_final.round()
            df_final.to_csv(os.path.join(dir_path,"detections_all.txt"),sep=' ')

        else:
            l = 1
            label = ["Camera1"]
            camera_totals = np.zeros((l,len(class_names)))
            df_final = totals_all[0]
            df_final = df_final.round()
            df_final.to_csv(os.path.join(dir_path,"detections_all.txt"),sep=' ')


        for n in range(l):
            for i, class_ in enumerate(class_names):
                camera_totals[n,i] = sum(totals_all[n][class_])


        camera_totals = pd.DataFrame(data=camera_totals.astype(int),columns=class_names)
        camera_totals.index = label
        camera_totals.to_csv(os.path.join(dir_path,"totals.txt"),index=True,sep=' ')            
