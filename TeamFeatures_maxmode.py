import skinmap as sm
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from scipy import ndimage, fftpack, stats
from skimage import exposure, measure
import pandas as pd
from PIL import Image
import cStringIO
import urllib2
import numpy as np
from pylab import *

def stdev_filter(arr, radius):
    """ Local standard deviation filter   """
    c1 = ndimage.filters.uniform_filter(arr, radius*2, mode='constant', origin=-radius)
    c2 = ndimage.filters.uniform_filter(arr*arr, radius*2, mode='constant', origin=-radius)
    return ((c2 - c1*c1)**.5)

def change_morphology_label(img):
    struct1 = ndimage.generate_binary_structure(2, 2)
    img_dilated = ndimage.morphology.binary_dilation(img, iterations=1, structure=struct1) #Food_Features_1.csv People_features_1.csv
    img_filled = ndimage.morphology.binary_fill_holes(img_dilated) #Food_Features_1.csv People_features_1.csv
    #img_filled = ndimage.morphology.binary_fill_holes(img) #2
    img_eroded = ndimage.morphology.binary_erosion(img_filled, iterations = 2, structure = struct1) #Food_Features_1.csv People_features_1.csv
    img_gaussian = ndimage.gaussian_filter(img_eroded,1) #Food_Features_1.csv People_features_1.csv
    #img_gaussian = ndimage.gaussian_filter(img_filled,1) #2
    
    labeled, shapes = ndimage.label(img_gaussian > 0)
    return labeled, shapes

def get_features(labeled, prop_img, img, shapes, img_size, img_texture):
    density_array = np.array([])
    area_covered = np.array([])
    hwratio = np.array([])
    density_hull = np.array([])
    circularity = np.array([])
    textures_array = np.array([])
    for label in range(1, shapes+1):
        density = float(((labeled==label)*img).sum())/float((labeled==label).sum())
        #if density>0.5 and density<1 and prop_img[label-1].area > 50: #Food_Features_1.csv People_features_1.csv
        if density<1 and prop_img[label-1].area > 50: #Food_Features_2.csv People_features_2.csv
            density_array=np.append(density_array, density)
            area_covered=np.append(area_covered, prop_img[label-1].area/img_size)
            hwratio=np.append(hwratio, prop_img[label-1].major_axis_length/prop_img[label-1].minor_axis_length)
            density_hull=np.append(density_hull, prop_img[label-1].solidity)
            circularity=np.append(circularity, prop_img[label-1].eccentricity)
            textures_array=np.append(textures_array, float(((labeled==label)*img_texture).sum())/float((labeled==label).sum()))

    if len(density_array) == 0:
        features = [len(density_array), -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    else:
        features = [len(density_array), max(area_covered), np.median(area_covered), max(density_array), np.median(density_array),\
                    max(hwratio), np.median(hwratio), max(density_hull), np.median(density_hull), max(circularity), np.median(circularity),max(textures_array), np.median(textures_array)]
    return features

#Food_Features_1.csv People_features_1.csv
#labels_df = pd.DataFrame(columns=['percent_skin_abg','percent_skin_cbcr','percent_skin_cccm',\
#                                      'percent_skin_equalized_abg','percent_skin_equalized_cbcr','percent_skin_equalized_cccm',\
#                                      'number_labels_abg', 'max_area_covered_abg', 'median_area_covered_abg','max_density_abg', 'median_density_abg',\
#                                      'max_hwratio_abg', 'median_hwratio_abg','max_density_hull_abg', 'median_density_hull_abg',\
#                                      'max_circularity_abg', 'median_circularity_abg',\
#                                      'number_labels_cbcr', 'max_area_covered_cbcr', 'median_area_covered_cbcr','max_density_cbcr', 'median_density_cbcr',\
#                                      'max_hwratio_cbcr', 'median_hwratio_cbcr','max_density_hull_cbcr', 'median_density_hull_cbcr',\
#                                      'max_circularity_cbcr', 'median_circularity_cbcr',\
#                                      'number_labels_cccm', 'max_area_covered_cccm', 'median_area_covered_cccm','max_density_cccm', 'median_density_cccm',\
#                                      'max_hwratio_cccm', 'median_hwratio_cccm','max_density_hull_cccm', 'median_density_hull_cccm',\
#                                      'max_circularity_cccm', 'median_circularity_cccm'])

labels_df = pd.DataFrame(columns=['percent_skin_equalized_cbcr','percent_skin_equalized_cbcr_cccm',\
                                      'number_labels_cbcr', 'max_area_covered_cbcr', 'median_area_covered_cbcr','max_density_cbcr', 'median_density_cbcr',\
                                      'max_hwratio_cbcr', 'median_hwratio_cbcr','max_density_hull_cbcr', 'median_density_hull_cbcr',\
                                      'max_circularity_cbcr', 'median_circularity_cbcr','max_texture_cbcr', 'median_texture_cbcr',\
                                      'number_labels_cbcr_cccm', 'max_area_covered_cbcr_cccm', 'median_area_covered_cbcr_cccm','max_density_cbcr_cccm', 'median_density_cbcr_cccm',\
                                      'max_hwratio_cbcr_cccm', 'median_hwratio_cbcr_cccm','max_density_hull_cbcr_cccm', 'median_density_hull_cbcr_cccm',\
                                      'max_circularity_cbcr_cccm', 'median_circularity_cbcr_cccm', 'max_texture_cbcr_cccm', 'median_texture_cbcr_cccm'])

#get url
file='People_MP.txt'
urls=np.loadtxt(file,dtype="str")
for url in urls:

    print url
    try:
        read= urllib2.urlopen(url).read()
    except urllib2.URLError:
        continue
    obj = Image.open( cStringIO.StringIO(read) )
    img= np.array(obj)
    #plt.imshow(img)
    #plt.show()

    obj_gray=np.array(obj.convert('L'))

    data= sm.SetUpImage(img)
    #print "image read in"

    #image_abg = data.ImgReg.skin.abg #Food_Features_1.csv People_features_1.csv
    #image_cbcr = data.ImgReg.skin.cbcr
    image_cccm = data.ImgReg.skin.cccm
    #image_cbcr_cccm = image_cbcr*image_cccm #2

    #image_EA_abg = data.ImgEqAdapt.skin.abg #Food_Features_1.csv People_features_1.csv
    image_EA_cbcr = data.ImgEqAdapt.skin.cbcr
    #image_EA_cccm = data.ImgEqAdapt.skin.cccm #2
    image_EA_cbcr_cccm = image_EA_cbcr*image_cccm #2

    texture = stdev_filter(obj_gray, 3) #2

    #labeled_abg, shapes_abg = change_morphology_label(image_EA_abg) #Food_Features_1.csv People_features_1.csv
    labeled_cbcr, shapes_cbcr = change_morphology_label(image_EA_cbcr)
    #labeled_cccm, shapes_cccm = change_morphology_label(image_EA_cccm) #Food_Features_1.csv People_features_1.csv
    labeled_cbcr_cccm, shapes_cbcr_cccm = change_morphology_label(image_EA_cbcr_cccm) #2

    #props_abg = measure.regionprops(labeled_abg) #Food_Features_1.csv People_features_1.csv
    props_cbcr = measure.regionprops(labeled_cbcr)
    #props_cccm = measure.regionprops(labeled_cccm) #Food_Features_1.csv People_features_1.csv
    props_cbcr_cccm = measure.regionprops(labeled_cbcr_cccm) #2
    #img_size = (labeled_abg.shape)[0]*(labeled_abg.shape)[1] #Food_Features_1.csv People_features_1.csv
    img_size = (labeled_cbcr.shape)[0]*(labeled_cbcr.shape)[1]

    #features_abg = get_features(labeled_abg, props_abg, image_EA_abg, shapes_abg, img_size) #Food_Features_1.csv People_features_1.csv
    features_cbcr = get_features(labeled_cbcr, props_cbcr, image_EA_cbcr, shapes_cbcr, img_size, texture)
    #features_cccm = get_features(labeled_cccm, props_cccm, image_EA_cccm, shapes_cccm, img_size) #Food_Features_1.csv People_features_1.csv
    features_cbcr_cccm = get_features(labeled_cbcr_cccm, props_cbcr_cccm, image_EA_cbcr_cccm, shapes_cbcr_cccm, img_size, texture)

    print "measure.regionprops done"

#Food_Features_1.csv People_features_1.csv
#    labels_df.loc[len(labels_df.index)]=([sm.percent_skin(image_abg), sm.percent_skin(image_cbcr), sm.percent_skin(image_cccm),\
#                                      sm.percent_skin(image_EA_abg), sm.percent_skin(image_EA_cbcr), sm.percent_skin(image_EA_cccm),\
#                                      features_abg[0], features_abg[1], features_abg[2], features_abg[3], features_abg[4], features_abg[5],\
#                                      features_abg[6], features_abg[7], features_abg[8], features_abg[9], features_abg[10],\
#                                      features_cbcr[0], features_cbcr[1], features_cbcr[2], features_cbcr[3], features_cbcr[4], features_cbcr[5],\
#                                      features_cbcr[6], features_cbcr[7], features_cbcr[8], features_cbcr[9], features_cbcr[10],\
#                                      features_cccm[0], features_cccm[1], features_cccm[2], features_cccm[3], features_cccm[4], features_cccm[5],\
#                                      features_cccm[6], features_cccm[7], features_cccm[8], features_cccm[9], features_cccm[10]])

#Food_Features_2.csv People_features_2.csv
    labels_df.loc[len(labels_df.index)]=([sm.percent_skin(image_EA_cbcr), sm.percent_skin(image_EA_cbcr_cccm),\
                                      features_cbcr[0], features_cbcr[1], features_cbcr[2], features_cbcr[3], features_cbcr[4], features_cbcr[5],\
                                      features_cbcr[6], features_cbcr[7], features_cbcr[8], features_cbcr[9], features_cbcr[10],\
                                      features_cbcr[11],features_cbcr[12],\
                                      features_cbcr_cccm[0], features_cbcr_cccm[1], features_cbcr_cccm[2], features_cbcr_cccm[3], features_cbcr_cccm[4], features_cbcr_cccm[5],\
                                      features_cbcr_cccm[6], features_cbcr_cccm[7], features_cbcr_cccm[8], features_cbcr_cccm[9], features_cbcr_cccm[10],\
                                      features_cbcr_cccm[11],features_cbcr_cccm[12]])

    #print "label features added to data frame"

labels_df.to_csv("People_Features_3.csv")
