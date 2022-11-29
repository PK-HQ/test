#!/usr/bin/env python#                      

import os                            
import numpy as np                                   
import pandas as pd
import nibabel as nb
import numpy.linalg as npl
import csv
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plot
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from scipy import stats
from scipy.stats.stats import pearsonr, ttest_1samp, percentileofscore,linregress






#PARAMETERS
cog_function="memory" #insert Neurosynth term, e.g. "memory" from the Neurosynth map 'memory_pFgA_z_FDR_0.01.nii'
ROI="LCx" #default options: 'LCx' or 'LSCx' 
Ngenes=15626 #default options: 1 to 15626
samples=1285+2 #497 1285

#FILE PATHS
dir_path = "filepath/"
input_path = os.path.join(dir_path, "input/")
output_path = os.path.join(dir_path,"output/")

#OUTPUT FILE
output_file = (os.path.join(output_path,"Data - "+cog_function+"_"+ROI+".csv")) #full data output
output_file_GSEA = (os.path.join(output_path, cog_function+"_"+ROI+".txt")) # .txt input file for GSEA (converted to .rnk later)

#INPUT FILES
#skiprows=1,nrows=0:15626,usecols=2:1623

#Allen Human Brain Atlas inputs
### Gene expr file name
print "Loading input files..."
#AHBA_file="aurinaGeneExprNorm"+ROI+".csv" #aurinaGeneExprNormLSCx.csv
AHBA_file="AHBA_16906_genes_normalized_1.csv" #aurinaGeneExprNormLSCx.csv

### Gene expr
AHBA_data = pd.read_csv(input_path+AHBA_file, usecols=list(range(2,samples)),nrows=Ngenes) #use columns 2-3704 (or should be 1-3704?)
#print(AHBA_data.head(5))
AHBA_data_zscore = pd.read_csv(input_path+AHBA_file, usecols=list(range(2,samples)),nrows=1)
#print(AHBA_data_zscore.head(5))
### Gene symbols
AHBA_genesymbols= pd.read_csv(input_path+AHBA_file, usecols=list(range(0,1))) #gene symbols for 16906 gene-probe mappings
AHBA_genesymbols=AHBA_genesymbols['geneSymbol']
#print(AHBA_genesymbols.head(5))
### DonorIDs aurinaDonoridLSCx.csv
AHBA_donorids = pd.read_csv(input_path+"aurinaDonorid"+ROI+".csv") #AurinaDonorIDsLCx.csv
AHBA_donorids=AHBA_donorids['donors'].tolist()
AHBA_donorids_ROI = map(unicode, AHBA_donorids) #AHBA donor ids of datapoints for region of interest
#print(AHBA_donorids_ROI.head(5))
### Sample MNI aurinaMNICoordLSCX
#aurinaMNICoords=pd.read_csv(input_path+"aurinaMNICoord"+ROI+".csv", usecols=list(range(1,4))) #AurinaMNICoordinatesLCx.csv



AHBA_wellids=pd.read_csv(input_path+"AHBA_wellids"+ROI+".csv") #well ids for sampling sites on donor brains
AHBA_wellids=AHBA_wellids['wellids'].tolist()
AHBA_index_ROI = pd.read_csv(input_path+"index_"+ROI+".csv") #index to extract desired sampling site datapoints for region of interest 
AHBA_index_ROI = AHBA_index_ROI['index'].tolist()




#Neurosynth inputs
mni_coordinates_file=os.path.join(input_path+"aurinaMNICoord"+ROI+".csv") #corrected MNI co-ordinates for co-registration
#aurinaMNICoords=pd.read_csv(input_path+"aurinaMNICoord"+ROI+".csv", header=0, usecols=list(range(1,4)))
Neurosynth_map = os.path.join(input_path,cog_function+"_pFgA_z_FDR_0.01.nii.gz") #map of cognitive function of interest
#print Neurosynth_map
mask = []
radius = 6 #for smoothing match




#DEFINING ANALYSIS FUNCTION
def get_mni_coordinates_from_wells(AHBA_wellids_ROI):
    """ #Spatial co-registration of AIBS Brain Institute Transcriptome
     and fMRI Statistical Map data (from Neurosynth)"""
    frame = pd.read_csv(mni_coordinates_file, header=0, index_col=0)
    #print frame.shape
    return list(frame.ix[AHBA_wellids_ROI].itertuples(index=False))        

def get_sphere(coords, r, vox_dims, dims):
    """ # Return all points within r mm of coordinates. Generates a cube
    and then discards all points outside sphere. Only returns values that
    fall within the dimensions of the image."""
    r = float(r)
    xx, yy, zz = [slice(-r / vox_dims[i], r / vox_dims[
                        i] + 0.01, 1) for i in range(len(coords))]
    cube = np.vstack([row.ravel() for row in np.mgrid[xx, yy, zz]])
    sphere = cube[:, np.sum(np.dot(np.diag(
        vox_dims), cube) ** 2, 0) ** .5 <= r]
    sphere = np.round(sphere.T + coords)
    return sphere[(np.min(sphere, 1) >= 0) & (np.max(np.subtract(sphere, dims), 1) <= -1),:].astype(int)

def get_values_at_locations(nifti_file, locations, radius, mask_file=None,  verbose=False):
    """ #Extract fMRImap zscore values """
    values = []
    nii = nb.load(nifti_file)
    data = nii.get_data()
    #print nii.shape
    if verbose:
        print "No mask provided - using implicit (not NaN, not zero) mask"
    mask = np.logical_and(np.logical_not(np.isnan(data)), data != 0)



    for location in locations:
        #print "Locations dimension %s " % location
        
        #print location
        #print npl.inv(nii.get_affine()), location
        #print nb.affines.apply_affine(npl.inv(nii.get_affine()), location)

        coord_data = [round(i) for i in nb.affines.apply_affine(npl.inv(nii.get_affine()), location)]
        sph_mask = (np.zeros(mask.shape) == True)
        if radius:
            sph = tuple(get_sphere(coord_data, vox_dims=nii.get_header().get_zooms(),r=radius, dims=nii.shape).T)
            sph_mask[sph] = True
        else:
            #if radius is not set use a single point
            sph_mask[coord_data[0], coord_data[1], coord_data[2]] = True
        roi = np.logical_and(mask, sph_mask)
        #If the roi is outside of the mask it is treated as a missing value
        if np.any(roi):
            val = data[roi].mean()
        else:
            val = np.nan
        values.append(val)
    return values

def approximate_random_effects(data, labels, group):
    slope_per_donor = np.array([])
    rval_per_donor = np.array([])
    #print "Performing approximate random effect analysis..."                    
    for donor_id in set(data[group]): #for donor_id in donorids, perform linear regression
        #print "Total usable datapoints of donor %s: %d" % (donor_id, len(list(data[labels[0]][data[group] == donor_id]))) #shows usable datapoints per donor
        slope, _, rval, p_val, stderr = linregress(list(data[labels[0]][data[group] == donor_id]),
                                                       list(data[labels[1]][data[group] == donor_id]))
        slope_per_donor=np.append(slope_per_donor,slope)
        rval_per_donor=np.append(rval_per_donor,rval)



    #average_slope = round(slope_per_donor.mean(),6) #get mean r-value across donors
    #average_rval = round(rval_per_donor.mean(),6) #get mean r-value across donors
    average_slope = round(np.nanmean(slope_per_donor),6) #get mean r-value across donors
    average_rval = round(np.nanmean(rval_per_donor),6) #get mean r-value across donors
    t_value, p_value = ttest_1samp(slope_per_donor, 0) #t-test (redundant information for downstream analyses)
    with open(output_file,'a') as f: #saving full data to .csv
        w = csv.writer(f)
        #print "Saving the analysis results..."
        w.writerow([gene,average_rval,average_slope,rval_per_donor[0],rval_per_donor[1],rval_per_donor[2],rval_per_donor[3],rval_per_donor[4],rval_per_donor[5],t_value,p_value])

    with open(output_file_GSEA,'a') as f: #saving GSEA input data to .csv
        w = csv.writer(f,delimiter = '\t')
        #print "Saving to GSEA input file..."
        w.writerow([gene,average_rval])

    #Scatterplot of gene expression against reverse inference fMRI map z-score
    print "Plotting the correlation graph..."
    ax=sns.lmplot(labels[0], labels[1], data, hue=group, legend=True, fit_reg=True) #comment-out for no plotting
    ax.set(xlabel="%s map z-score value" % (cog_function.capitalize()))
    ax = plot.title(gene)
    print "Saving the correlation graph..."
    plot.savefig(plot_pdf, format='pdf')
    plot.close()
    return

with open(output_file,'w') as f: #column headers for full data .csv file
    w = csv.writer(f)
    w.writerow(['gene symbol','mean r-value','mean slope','H0351.2001 r-value','H0351.1012 r-value','H0351.2002 r-value','H0351.1016 r-value','H0351.1015 r-value','H0351.1009 r-value','t','p'])




#ANALYSIS
plot_pdf = PdfPages(os.path.join(output_path,"Plot - "+cog_function+"_"+ROI+".pdf"))



#GET NEUROSYNTH Z-SCORES
print "Extracting Neurosynth data for region of interest (mask)..."
for (gene),(index, expr_values) in zip(AHBA_genesymbols,AHBA_data_zscore.iterrows()):
    expr_values_ROI = list(expr_values)
    #print len(AHBA_wellids)
    #print len(AHBA_index_ROI)    
    AHBA_wellids_ROI = [AHBA_wellids[i] for i in AHBA_index_ROI]                                                  
    #expr_values_ROI = [expr_values[i] for i in AHBA_index_ROI]
    print "Getting MNI space coordinates"                         
    mni_coordinates = get_mni_coordinates_from_wells(AHBA_wellids_ROI)                      
    #mni_coordinates = list(aurinaMNICoords)
    print "Checking values of the provided NIFTI file at well locations"
    #print mni_coordinates
    fMRImap_zscores = get_values_at_locations(                                     
    Neurosynth_map, mni_coordinates, mask_file=mask, radius=radius, verbose=True)  #extracting neurosynth data as mask
    fMRImap_zscores=[np.nan if x < 0 else x for x in fMRImap_zscores]
    print "Co-registration done and saved!"        

#SPATIAL CORRELATION ANALYSIS
print "Starting analysis for region of interest..."
for (gene),(index, expr_values) in zip(AHBA_genesymbols,AHBA_data.iterrows()): #removed NHLRC1, gene 14422
    index=index+1
    print "Gene #%d" % index
    expr_values_ROI = list(expr_values)                                                 
    #expr_values_ROI = [expr_values[i] for i in AHBA_index_ROI]
    #expr_values_ROI=np.array(expr_values_ROI)
    #expr_values_ROI=stats.zscore(expr_values_ROI, axis=0, ddof=1)
    names = ["NIFTI values", "%s expression" % gene, "donor ID"] # preparing the data frame
    #print len(fMRImap_zscores)
    #print len(expr_values_ROI)
    #print len(AHBA_donorids_ROI)
    data = pd.DataFrame(np.array([fMRImap_zscores, expr_values_ROI, AHBA_donorids_ROI]).T, columns=names)
    data.ix[:,0:2]=data.ix[:,0:2].apply(pd.to_numeric, errors='coerce')
    len_before = len(data) #number of neurosynth datapoints
    data.dropna(axis=0, inplace=True)
    #nans = len_before - len(data)
    #if nans > 0:
        #print "%s of %s AHBA %s wellids fall outside of the Neurosynth %s mask" % (nans,len_before,ROI.replace("_", " "),cog_function)
    approximate_random_effects(data, ["NIFTI values", "%s expression" % gene], "donor ID") #analysis
plot_pdf.close() #comment-out for no plotting
# main()


#HOW TO PREPARE GSEA-PRERANKED INPUT FILE
'''txt_file = output_file_GSEA
base = os.path.splitext(txt_file)[0]
os.rename(txt_file, base + ".rnk") #converting input file to .rnk format for GSEA preranked'''