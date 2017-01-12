import os
import datetime
import random
import shutil
import cv2


class Utils(object):
	def __init__(self):
		pass
		

	def processVideoXMLFolder(self, videoFolder, xmlfolder, pathToSave):
		for f in os.listdir(videoFolder):
			filename, fileextension = os.path.splitext(f)
			videofile = os.path.join(videoFolder, f)
			xmlfile = os.path.join(xmlfolder, filename+'.xml')
			if (os.path.isfile(videofile) and os.path.isfile(xmlfile)):
				print("\nProcssing {0}:".format(filename))
				self.processVideoXMLPair(filename, videoFolder, xmlfolder, pathToSave)
			else:
				print("\nVideo or xml not found for {0}:".format(f))
	


	def processVideoXMLPair(self, filename, videoPath, xmlPath, pathToSave):
		xmlfilename = filename + ".xml"
		videofilename = filename + ".MP4"
		xmldata = self.readXMLdata(os.path.join(xmlPath, xmlfilename))
		cap = cv2.VideoCapture(os.path.join(videoPath, videofilename))
		if(not cap.isOpened()):
			print("Could not open video {0}".format(videofilename))
			return

		imdumped = 0
		imskipped = 0
		for k,v in xmldata.iteritems():
			cap.set(1, int(k)) #CAP_PROP_POS_FRAMES=1
			ret,frame = cap.read()
			imheight, imwidth, imchannels = frame.shape
			if(ret==True):
				#cv2.imshow("Frame",frame)
				#cv2.waitKey(10)
				#print("Frame {0}:".format(k))
				for box in v:
					boxw = int(box['width'])
					boxh = int(box['height'])
					if(boxw < 36 or boxh < 72):
						imskipped = imskipped + 1
						continue
					numFiles = self.countFiles(pathToSave)
					y1 = int(box['y']) - boxh/2
					y2 = y1 + boxh
					x1 = int(box['x']) - boxw/2
					x2 = x1 + boxw
					xpad1 = int(random.uniform(boxw*0.0, boxw*0.10))
					ypad1 = int(random.uniform(boxh*0.0, boxh*0.10))
					xpad2 = int(random.uniform(boxw*0.0, boxw*0.10))
					ypad2 = int(random.uniform(boxh*0.0, boxh*0.10))

					x1 = x1-xpad1
					x2 = x2+xpad2
					if(x1 < 0): x1 = 0
					if(x2 > imwidth-1): x2 = imwidth-1

					y1 = y1-ypad1
					y2 = y2+ypad2
					if(y1 < 0): y1 = 0
					if(y2 > imheight-1): y2 = imheight-1

					roi = frame[y1:y2, x1:x2]
					imgresized = cv2.resize(roi,(36, 72), interpolation = cv2.INTER_CUBIC)
					flipped = cv2.flip(imgresized, 1)
					cv2.imwrite(pathToSave + "/" + self.getDate() + "_" + self.getTime() + "_" + str(numFiles+1) + ".jpg", imgresized)
					cv2.imwrite(pathToSave + "/" + self.getDate() + "_" + self.getTime() + "_" + str(numFiles+1) + "_F" + ".jpg", flipped)
					imdumped = imdumped + 1
					#print("x={0}, y={1}, width={2}, height={3}".format(box['x'], box['y'], box['width'], box['height']))
		
		print("Images processed = {0}, Images skipped = {1}".format(imdumped, imskipped))

		
	


	def readXMLdata(self, filePath):
		try:
			from lxml import etree
			print("running with lxml.etree")
			tree = etree.parse(filePath)
			root = tree.getroot()
			if(root.tag != 'opencv_storage'):
				print("Unexpected root tag")
				return None
			else:
				children = list(root)
				if(children[0].tag != 'Info' or children[1].tag != 'Test'):
					print("Unexpected children tag")
					return None
				info_ = list(children[0])[0]
				Version = None
				VideoClip = None
				for child in info_:
					if(child.tag == 'Version'):
						Version = child.text
					if(child.tag == 'VideoClip'):
						VideoClip = child.text
				

				if(Version != 'IVEMT_1.21'):
					print("Unexpected IVEMT version")
					return None
				print('Reading markup for clip {0}'.format(VideoClip) )


				Test = list(children[1]) #all the underscores under Test
				print('Number of frames marked = {0}'.format(len(Test)) )
				DataDictionary = {}
				for t in Test: #now take each underscores
					thisframedata = list(t) #all the children of underscore
					FrameIDComment = thisframedata[0] #first child is frame id comment
					underscoreagain = thisframedata[1] #this child keeps actual data, under sub-children
					
					FrameID = underscoreagain[0].text
					ObjectData = underscoreagain[1]
					#print("FrameID={0}".format(FrameID))
					DataDictionary[str(FrameID)] = []
					for obj_ in list(ObjectData):
							children = list(obj_)
							x=None
							y=None
							width = None
							height=None
							for c in children:
								if(c.tag == 'X'):
									x=c.text
								if(c.tag == 'Y'):
									y=c.text
								if(c.tag == 'Width'):
									width=c.text
								if(c.tag == 'Height'):
									height=c.text
							if(x==None or y==None or width==None or height==None):
								print("Unexpected x,y,width,height")
								return None
						
							box = {'x':x, 'y':y, 'width':width, 'height':height}
							DataDictionary[str(FrameID)].append(box)


				return DataDictionary
						








		except ImportError:
			print("Failed to import etree")
				



		
	def countFiles(self, path):
		try:
			return sum(os.path.isfile(os.path.join(path, f)) for f in os.listdir(path))
				
		except:
			if(~os.path.isdir(path)):
				print("Bad path {0}".format(path))
			else:
				print("Unknown Error in countFiles()")
				
				
	def getTime(self):
		now = datetime.datetime.now()
		date = str(now.hour) + "_" + str(now.minute) + "_" + str(now.second)
		return date

	def getDate(self):
		now = datetime.datetime.now()
		date = str(now.year) + "_" + str(now.month) + "_" + str(now.day)
		return date
		
	def getFileCountInDir(self, directory):
		print len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])
		

	def getAllFilesInDir(self, directory):
		files = []
		for f in os.listdir(directory):
			if os.path.isfile(os.path.join(directory, f)):
				if(f.endswith(".MP4")):
					files.append(os.path.join(directory, f))
				
		return files
		
	def moveRandomFiles(self, srcFolder, dstFolder, maxFilesToMove):
		count = 0
		for f in os.listdir(srcFolder):
			if os.path.isfile(os.path.join(srcFolder, f)):
				if random.uniform(0, 1) > 0.5:
					shutil.move(os.path.join(srcFolder, f), os.path.join(dstFolder, f))
					count = count + 1
				if(count >= maxFilesToMove): break
				
		return count
		
	
	def moveAllFiles(self, srcFolder, dstFolder):
		count = 0
		for f in os.listdir(srcFolder):
			if os.path.isfile(os.path.join(srcFolder, f)):
				shutil.move(os.path.join(srcFolder, f), os.path.join(dstFolder, f))
				count = count + 1
				
		return count
			
		
		
if __name__ == "__main__":
	utilsTest = Utils()
	# videofolder = '/home/ivusi7dl/Videos/ADASPedestrainClips_Nov26_2016'
	# xmlfolder = '/home/ivusi7dl/Videos/Markups/ADASPedestrainClips_Nov26_2016'
	# pathToSave = '/home/ivusi7dl/Videos/ADASPedestrainClips_Nov26_2016_Images'
	# utilsTest.processVideoXMLFolder(videofolder, xmlfolder, pathToSave)

	# srcFolder = "/home/ivusi7dl/Users/chandan/pycode/tf/cnn_pd/detections_dump"
	# dstFolder = "/home/ivusi7dl/Users/chandan/data/Nov23_2016/Validate/1"
	# num = utilsTest.moveRandomFiles(srcFolder, dstFolder, 200)
	
	srcFolder = "/home/ivusi7dl/Videos/ADASPedestrainClips_Nov26_2016_Images"
	# srcFolder = "/home/ivusi7dl/Users/chandan/pycode/tf/cnn_pd/detections_dump"
	dstFolder = "/home/ivusi7dl/Users/chandan/data/Nov23_2016/Train/1"
	num = utilsTest.moveAllFiles(srcFolder, dstFolder)
	
	print("Num files moved = {0}".format(num))
	print("Num files in {0} = {1}".format(srcFolder, utilsTest.getFileCountInDir(srcFolder)))
	print("Num files in {0} = {1}".format(dstFolder, utilsTest.getFileCountInDir(dstFolder)))
	
	#folderName = "/home/chandangope/Pictures"
	#numFiles = utilsTest.countFiles(folderName)
	#print("Num files in folder {0} = {1}".format(folderName, numFiles))
	#print("Current datetime {0}".format(utilsTest.getDate()))
	
	#folderName = "/media/chandangope/Expansion Drive/Clips/ADAS/LDW/RegressionTestClips/YesLDW"
	#files = utilsTest.getAllFilesInDir(folderName)
	#for f in files:
	#	print f
