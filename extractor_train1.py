#!C:\Python27\python.exe

import os
import sys
import glob
import time
from optparse import OptionParser
from SimpleCV import (
    random,
    Color,
    KNNClassifier,
    NaiveBayesClassifier,
    TreeClassifier,
    SVMClassifier,
    HueHistogramFeatureExtractor,
    EdgeHistogramFeatureExtractor,
    HaarLikeFeatureExtractor,
    BOFFeatureExtractor,
    Image,
    ImageSet
)
from os import listdir
from os.path import isfile, join

def gerador_Header():
    #Abrir arquivo que vai receber as features dos arquivos de imagens
    file = "image_features_train_1.arff"
    text_file = open(file, "w")
    text_file.write("@RELATION image_features" + "\n")
    text_file.write("\n")
    text_file.write("@ATTRIBUTE Edge_Lenght0" + " " + "REAL" + "\n")
    text_file.write("@ATTRIBUTE Edge_Angle0" + " " + "REAL" + "\n")
    
    #text_file.write("@ATTRIBUTE Hue0" + " " + "REAL" + "\n")
    #text_file.write("@ATTRIBUTE Hue1" + " " + "REAL" + "\n")
    #text_file.write("@ATTRIBUTE Hue2" + " " + "REAL" + "\n")
    #text_file.write("@ATTRIBUTE Hue3" + " " + "REAL" + "\n")
    #text_file.write("@ATTRIBUTE Hue4" + " " + "REAL" + "\n")
    #text_file.write("@ATTRIBUTE Hue5" + " " + "REAL" + "\n")
    #text_file.write("@ATTRIBUTE Hue6" + " " + "REAL" + "\n")
    #text_file.write("@ATTRIBUTE Hue7" + " " + "REAL" + "\n")
    #text_file.write("@ATTRIBUTE Hue8" + " " + "REAL" + "\n")
    #text_file.write("@ATTRIBUTE Hue9" + " " + "REAL" + "\n")
    text_file.write("@ATTRIBUTE class" + "  " + "{pistola, revolver}" + "\n")
    text_file.write("\n")
    text_file.write("@DATA" + "\n")
    text_file.close()

print ""
print ""
print "......Iniciando Extrator de Caracteristicas de Imagens"
print "......Classe 1"
print ""
print ""

def feature_extractor():

    #Abrir arquivo que vai receber as features dos arquivos de imagens
    file = "image_features_train_1.arff"
    text_file = open(file, "a")
    trainPath = "C:/Python27/ARP/Dataset/train2"
    dirList = os.listdir(trainPath)
    edge = EdgeHistogramFeatureExtractor(1)
    for dirName in dirList:
		fileList = os.listdir(trainPath + '/' + dirName)
		maxsize = len(fileList)
		for i in range(0, maxsize):
			#print fileList[i]
			filePATH = (trainPath + '/' + dirName + '/' + fileList[i])
			img = Image(filePATH)
			#Feature - Edge histogram
			edge_vecs = edge.extract(img)
			edge_fields = edge.getFieldNames()
			for y in range(0, edge.getNumFields()):
				#print edge_fields[y], '=', edge_vecs[y]
				text_file.write(str(edge_vecs[y]) + ",")
			#Feature - Hue	
			#hue = HueHistogramFeatureExtractor(10)
			#hue_vecs = hue.extract(img)
			#hue_fields = hue.getFieldNames()
			#for i in range(0, hue.getNumFields()):
				#print hue_fields[i], '=', hue_vecs[i]
				#text_file.write(str(hue_vecs[i]) + ",")
			text_file.write(dirName + "\n")
    text_file.close()
"""
onlyfiles = [f for f in listdir(trainPath) if isfile(join(mypath, f))]
    for dirName in dirList:
		if os.path.isdir(trainPath + '/' + dirName):
			#classes.append(dirName) 
			#text_file.write("Feature" + "," + "Valor" +"\n")
			#Feature - Edge histogram
			edge = EdgeHistogramFeatureExtractor(5)
			edge_vecs = edge.extract(img)
			edge_fields = edge.getFieldNames()
			for i in range(0, edge.getNumFields()):
				print edge_fields[i], '=', edge_vecs[i]
				text_file.write(str(edge_vecs[i]) + ",")
			#Feature - Hue	
			hue = HueHistogramFeatureExtractor(5)
			hue_vecs = hue.extract(img)
			hue_fields = hue.getFieldNames()
			for i in range(0, hue.getNumFields()):
				print hue_fields[i], '=', hue_vecs[i]
				text_file.write(str(hue_vecs[i]) + ",")
    #Feature - Haar
    #haar = HaarLikeFeatureExtractor(fname='haar.txt')
    #haar_vecs = haar.extract(img)
    #haar_fields = haar.getFieldNames()
    #for i in range(0, len(haar_fields)):
    #    print haar_fields[i], '=', haar_vecs[i]
    #    text_file.write(str(haar_vecs[i]) + ",")
    text_file.write("revolver")
    text_file.close()
"""
#Criando o header do arquivo ARFF de features	
var1 = gerador_Header()

#Gerando e gravando as features no arquivo ARFF
var2 = feature_extractor()
	
#classes = ['pistola_gray','revolver_gray',]

#img = Image('revolver.jpg')


	
#text_file.write("revolver")
#text_file.close()