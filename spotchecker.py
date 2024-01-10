
from math import sqrt
from skimage.feature import blob_log
import numpy as np
from itertools import product
import pandas as pd
import tifffile as tiff



# plot spots on the image.
# figure, axes = plt.subplots(figsize=(100, 100), sharex=True, sharey=True)
# axes.imshow(FRchannel,cmap ='binary')
# for blob in blobs_log_FR:
#     y, x, r = blob
#     c = plt.Circle((x, y), r, color='red', linewidth=1, fill=False)
#     axes.add_patch(c)
# plt.tight_layout()
# plt.savefig('Far-red-channel.png')
# plt.clf()
#
# figure, axes = plt.subplots(figsize=(100, 100), sharex=True, sharey=True)
# axes.imshow(GFPchannel,cmap ='binary')
# for blob in blobs_log_GFP:
#     y, x, r = blob
#     c = plt.Circle((x, y), r, color='red', linewidth=1, fill=False)
#     axes.add_patch(c)
# plt.tight_layout()
# plt.savefig('GFP-channel.png')
# plt.clf()
#
# figure, axes = plt.subplots(figsize=(100, 100), sharex=True, sharey=True)
# axes.imshow(RFPchannel,cmap ='binary')
# for blob in blobs_log_RFP:
#     y, x, r = blob
#     c = plt.Circle((x, y), r, color='red', linewidth=1, fill=False)
#     axes.add_patch(c)
# plt.tight_layout()
# plt.savefig('RFP-channel.png')
# plt.clf()
# #
# figure, axes = plt.subplots(figsize=(100, 100), sharex=True, sharey=True)
# axes.imshow(beads,cmap ='binary')
# for blob in blobs_log_beads:
#     y, x, r = blob
#     c = plt.Circle((x, y), r, color='red', linewidth=1, fill=False)
#     axes.add_patch(c)
# plt.tight_layout()
# plt.savefig('beads-channel.png')
# plt.clf()


def points_in_circle(radius):
    for x, y in product(range(int(radius) + 1), repeat=2):
        if x ** 2 + y ** 2 <= radius ** 2:
            yield from set(((x, y), (x, -y), (-x, y), (-x, -y),))


def blobmeangetter(blob,RFPchannel,GFPchannel,FRchannel,beads,option):
    if option == 3:
        #   makes the x y and r into a list
        listof = blob.tolist()
        coordinates = []
        #   gets all the relative values based on the radius
        operations = list(points_in_circle(listof[2]))
        #   turns the relative values into real values based on the x y pos
        for item in operations:
            newx = listof[0] + item[0]
            newy = listof[1] + item[1]
            if newx > 2025 or newy > 2025:
                pass
            else:
                coordinates.append([int(newx), int(newy)])
        #     current=coordinates[0]

        datafromblob = []

        for item in coordinates:
            xpos = item[0]
            ypos = item[1]
            r = RFPchannel[xpos, ypos]
            g = GFPchannel[xpos, ypos]
            fr = FRchannel[xpos, ypos]
            bd = beads[xpos, ypos]
            datafromblob.append([int(r), int(g), int(fr), int(bd)])

        df = pd.DataFrame.from_records(datafromblob)
        df.columns = ["Red", "Green", "FarRed", "Bead"]
        Rmean = df["Red"].mean()
        # print(Rmean)
        Gmean = df["Green"].mean()
        FRmean = df["FarRed"].mean()
        Bdmean = df["Bead"].mean()
        Xloc = blob[1]
        Yloc = blob[0]
        return (Rmean, Gmean, FRmean, Bdmean, Xloc, Yloc)
    if option == 2:
        #   makes the x y and r into a list
        listof = blob.tolist()
        coordinates = []
        #   gets all the relative values based on the radius
        operations = list(points_in_circle(listof[2]))
        #   turns the relative values into real values based on the x y pos
        for item in operations:
            newx = listof[0] + item[0]
            newy = listof[1] + item[1]
            if newx > 2025 or newy > 2025:
                pass
            else:
                coordinates.append([int(newx), int(newy)])
        #     current=coordinates[0]

        datafromblob = []

        for item in coordinates:
            xpos = item[0]
            ypos = item[1]
            r = RFPchannel[xpos, ypos]
            g = GFPchannel[xpos, ypos]
            bd = beads[xpos, ypos]
            datafromblob.append([int(r), int(g), int(bd)])

        df = pd.DataFrame.from_records(datafromblob)
        df.columns = ["Red", "Green", "Bead"]
        Rmean = df["Red"].mean()
        # print(Rmean)
        Gmean = df["Green"].mean()
        Bdmean = df["Bead"].mean()
        Xloc = blob[1]
        Yloc = blob[0]
        return (Rmean, Gmean, Bdmean, Xloc, Yloc)


def thresholder(dotframe,color,dataframe):
    realmean=dotframe[color].mean()
    backmean=np.mean(dataframe)
    output= backmean+((realmean-backmean)/4)
    return(output)

def VesicleType(array):
    variable=""
    if array['Rratio'] == True:
        variable=variable+"R"
    if array['Gratio'] == True:
        variable=variable+"G"
    try:
        if array['FRratio'] == True:
            variable=variable+"Fr"
    except:
        pass
    return(variable)


def percentageer(value,framesum):
    result=(value/framesum)*100
    return(result)




def graphmaker(name,dataset,RFPthreashold,GFPthreashold,FRthreashold,option):
    if option == 3:
        dataset.loc[dataset['Red'] <= RFPthreashold, 'Rratio'] = False
        dataset.loc[dataset['Red'] > RFPthreashold, 'Rratio'] = True
        dataset.loc[dataset['Green'] <= GFPthreashold, 'Gratio'] = False
        dataset.loc[dataset['Green'] > GFPthreashold, 'Gratio'] = True
        dataset.loc[dataset['FarRed'] <= FRthreashold, 'FRratio'] = False
        dataset.loc[dataset['FarRed'] > FRthreashold, 'FRratio'] = True

        dataset['Type'] = dataset.apply(VesicleType, axis=1)

        countsdata = dataset['Type'].value_counts()
        # print("-----------------")
        # print(name+" counts")
        # print(countsdata)
        # print("-----------------")
        # print(type(countsdata))
        countsum=countsdata.sum()
        percentages = countsdata.apply(lambda x: (x/countsum)*100)
        # print(type(countsdata))
        return(countsdata)
    if option == 2:
        dataset.loc[dataset['Red'] <= RFPthreashold, 'Rratio'] = False
        dataset.loc[dataset['Red'] > RFPthreashold, 'Rratio'] = True
        dataset.loc[dataset['Green'] <= GFPthreashold, 'Gratio'] = False
        dataset.loc[dataset['Green'] > GFPthreashold, 'Gratio'] = True

        dataset['Type'] = dataset.apply(VesicleType, axis=1)

        countsdata = dataset['Type'].value_counts()
        # print("-----------------")
        # print(name+" counts")
        # print(countsdata)
        # print("-----------------")
        # print(type(countsdata))
        countsum = countsdata.sum()
        percentages = countsdata.apply(lambda x: (x / countsum) * 100)
        # print(type(countsdata))
        return (countsdata)

def datamaker(option):
    inputfile='control_after.tiff'
    # test background flourescene from image then set
    # compare dots of chanels within centroid distance
    RFPthreshold=.001
    GFPthreshold=.0008
    if option == 3:
        FRthreashold=.0009
    beadsthreshold=.005

    img = tiff.imread(inputfile)
    imarray = np.array(img)
    imarray2 = imarray[10:2037, 10:2037]
    # img1=imarray[0, :, :, :, 0]
    GFPchannel=imarray2[:, :, 0]
    RFPchannel=imarray2[:, :, 1]
    if option == 3:
        FRchannel=imarray2[:, :, 2]
    beads=imarray2[:, :, 3]

    after = np.zeros((beads.shape[0], beads.shape[1], 4), dtype=np.uint16)
    if option == 3:
        after[:, :, 0] = FRchannel
    after[:, :, 1] = GFPchannel
    after[:, :, 2] = RFPchannel
    after[:, :, 3] = beads

    tiff.imsave('dcgcheck2.tiff', after)

    # f, axarr = plt.subplots(1,3,figsize=(15,15))

    # axarr[0].imshow(greens,cmap ='Greens')
    # axarr[1].imshow(reds,cmap ='Reds')
    # axarr[2].imshow(blues,cmap ='Blues')
    # axarr[3].imshow(farreds,cmap ='Magenta')


    if option == 3:
        blobs_log_FR = blob_log(FRchannel, max_sigma=4, num_sigma=8, threshold=FRthreashold)
    blobs_log_GFP = blob_log(GFPchannel, max_sigma=4, min_sigma=2, num_sigma=8, threshold=GFPthreshold)
    blobs_log_RFP = blob_log(RFPchannel, max_sigma=4, min_sigma=2, num_sigma=8, threshold=RFPthreshold)
    blobs_log_beads = blob_log(beads, max_sigma=4, min_sigma=2, num_sigma=8, threshold=beadsthreshold)
    print("datamaker1")
    # Compute radii in the 3rd column.
    if option == 3:
        blobs_log_FR[:, 2] = blobs_log_FR[:, 2] * sqrt(2)
    blobs_log_GFP[:, 2] = blobs_log_GFP[:, 2] * sqrt(2)
    blobs_log_RFP[:, 2] = blobs_log_RFP[:, 2] * sqrt(2)
    blobs_log_beads[:, 2] = blobs_log_beads[:, 2] * sqrt(2)

    listofRFPblobs = []
    listofGFPblobs = []
    if option == 3:
        listofFRblobs = []
    listofBeadblobs = []

    print("datamaker2")
    for item in blobs_log_RFP:
        if option == 3:
            listofRFPblobs.append(blobmeangetter(item,RFPchannel,GFPchannel,FRchannel,beads,option))
        if option == 2:
            FRchannel = "blank"
            listofRFPblobs.append(blobmeangetter(item, RFPchannel, GFPchannel,FRchannel, beads,option))

    for item in blobs_log_GFP:
        if option == 3:
            listofGFPblobs.append(blobmeangetter(item,RFPchannel,GFPchannel,FRchannel,beads,option))
        if option == 2:
            FRchannel = "blank"
            listofGFPblobs.append(blobmeangetter(item, RFPchannel, GFPchannel,FRchannel, beads,option))

    if option == 3:
        for item in blobs_log_FR:
            listofFRblobs.append(blobmeangetter(item,RFPchannel,GFPchannel,FRchannel,beads,option))

    for item in blobs_log_beads:
        if option == 3:
            listofBeadblobs.append(blobmeangetter(item,RFPchannel,GFPchannel,FRchannel,beads,option))
        if option == 2:
            FRchannel = "blank"
            listofBeadblobs.append(blobmeangetter(item,RFPchannel,GFPchannel,FRchannel,beads,option))

    dfGFP = pd.DataFrame.from_records(listofGFPblobs)
    if option == 3:
        dfGFP.columns = ["Red", "Green", "FarRed", "Bead", "Xloc", "Yloc"]
    if option == 2:
        dfGFP.columns = ["Red", "Green", "Bead", "Xloc", "Yloc"]
    print("datamaker3")

    dfRFP = pd.DataFrame.from_records(listofRFPblobs)
    if option == 3:
        dfRFP.columns = ["Red", "Green", "FarRed", "Bead", "Xloc", "Yloc"]
    if option == 2:
        dfRFP.columns = ["Red", "Green", "Bead", "Xloc", "Yloc"]

    if option == 3:
        dfFR = pd.DataFrame.from_records(listofFRblobs)
        dfFR.columns = ["Red", "Green", "FarRed", "Bead", "Xloc", "Yloc"]

    dfBeads = pd.DataFrame.from_records(listofBeadblobs)
    if option == 3:
        dfBeads.columns = ["Red", "Green", "FarRed", "Bead", "Xloc", "Yloc"]
    if option == 2:
        dfBeads.columns = ["Red", "Green", "Bead", "Xloc", "Yloc"]
    Beadrealdotmean = dfBeads["Bead"].mean()
    GFPrealdotmean = dfGFP["Green"].mean()
    RFPrealdotmean = dfRFP["Red"].mean()
    if option == 3:
        FRrealdotmean = dfFR["FarRed"].mean()

    dfBeads = dfBeads.sort_values(by=['Bead'])

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        dfBeads = dfBeads.reset_index(drop=True)

    beadvalue = dfBeads.loc[0, 'Bead'] - 50

    # 1450 is farred threashold
    # This removes the beads
    dfRFP_beads = dfRFP[dfRFP["Bead"].between(0, beadvalue)]
    # this deletes the bead column
    dfRFP_beads = dfRFP_beads.drop(["Bead"], axis=1)

    dfGFP_beads = dfGFP[dfGFP["Bead"].between(0, beadvalue)]
    dfGFP_beads = dfGFP_beads.drop(["Bead"], axis=1)
    if option == 3:
        dfFR_beads = dfFR[dfFR["Bead"].between(0, beadvalue)]
        dfFR_beads = dfFR_beads.drop(["Bead"], axis=1)
    print("datamaker4")
    GFPthreashold = thresholder(dfGFP_beads, "Green", GFPchannel)
    RFPthreashold = thresholder(dfRFP_beads, "Red", RFPchannel)
    if option == 3:
        FRthreashold = thresholder(dfFR_beads, "FarRed", FRchannel)
    print("datamaker5")
    if option == 3:
        countsdataG=graphmaker("GFP",dfGFP_beads,RFPthreashold,GFPthreashold,FRthreashold,option)
    if option == 2:
        FRthreashold = "blank"
        countsdataG=graphmaker("GFP",dfGFP_beads,RFPthreashold,GFPthreashold,FRthreashold,option)
    countsdataG1 = countsdataG.drop(labels=[''],errors='ignore')
    if option == 3:
        countsdataG1 = countsdataG1.drop(labels=['Fr'], errors='ignore')
    countsdataG1 = countsdataG1.drop(labels=['R'], errors='ignore')
    countsdataG1 = countsdataG1.drop(labels=['RFr'], errors='ignore')
    print(countsdataG1)
    print("datamaker6")
    if option == 3:
        countsdataR=graphmaker("RFP",dfRFP_beads,RFPthreashold,GFPthreashold,FRthreashold,option)
    if option == 2:
        FRthreashold= "blank"
        countsdataR=graphmaker("RFP",dfRFP_beads,RFPthreashold,GFPthreashold,FRthreashold,option)
    countsdataR1 = countsdataR.drop(labels=[''],errors='ignore')
    if option == 3:
        countsdataR1 = countsdataR1.drop(labels=['Fr'], errors='ignore')
    countsdataR1 = countsdataR1.drop(labels=['G'], errors='ignore')
    countsdataR1 = countsdataR1.drop(labels=['GFr'], errors='ignore')
    print(countsdataR1)

    print("datamaker7")
    print(option)
    if option == 3:
        print("test132123")
        countsdataFr=graphmaker("Far-Red",dfFR_beads,RFPthreashold,GFPthreashold,FRthreashold,option)
        countsdataFr1 = countsdataFr.drop(labels=[''],errors='ignore')
        countsdataFr1 = countsdataFr1.drop(labels=['R'], errors='ignore')
        countsdataFr1 = countsdataFr1.drop(labels=['G'], errors='ignore')
        countsdataFr1 = countsdataFr1.drop(labels=['RG'], errors='ignore')
        print(countsdataFr1)
    print("datamaker523524")
    #  calculations
    try:
        G = countsdataG1.at['G']
    except:
        G = 0
    try:
        if countsdataG1.at['RG'] == 0 and countsdataR1.at['RG'] != 0:
            RG = countsdataR1.at['RG']
        if countsdataR1.at['RG'] == 0 and countsdataG1.at['RG'] != 0:
            RG = countsdataG1.at['RG']
        if countsdataG1.at['RG'] != 0 and countsdataR1.at['RG'] != 0:
            RG = (countsdataG1.at['RG']+countsdataR1.at['RG'])/2
        if countsdataG1.at['RG'] == 0 and countsdataR1.at['RG'] == 0:
            RG = 0
    except:
        RG = 0
    if option == 3:
        try:
            if countsdataG1.at['RGFr'] == 0 and countsdataR1.at['RGFr'] != 0 and countsdataFr1.at['RGFr'] != 0:
                RGFr = (countsdataR1.at['RGFr']+countsdataFr1.at['RGFr'])/2
            if countsdataG1.at['RGFr'] != 0 and countsdataR1.at['RGFr'] == 0 and countsdataFr1.at['RGFr'] != 0:
                RGFr = (countsdataG1.at['RGFr']+countsdataFr1.at['RGFr'])/2
            if countsdataG1.at['RGFr'] != 0 and countsdataR1.at['RGFr'] != 0 and countsdataFr1.at['RGFr'] == 0:
                RGFr = (countsdataG1.at['RGFr']+countsdataR1.at['RGFr'])/2
            if countsdataG1.at['RGFr'] != 0 and countsdataR1.at['RGFr'] != 0 and countsdataFr1.at['RGFr'] != 0:
                RGFr = (countsdataG1.at['RGFr'] + countsdataR1.at['RGFr'] + countsdataFr1.at['RGFr']) / 3
            if countsdataG1.at['RGFr'] == 0 and countsdataR1.at['RGFr'] == 0 and countsdataFr1.at['RGFr'] != 0:
                RGFr = countsdataFr1.at['RGFr']
            if countsdataG1.at['RGFr'] == 0 and countsdataR1.at['RGFr'] != 0 and countsdataFr1.at['RGFr'] == 0:
                RGFr = countsdataR1.at['RGFr']
            if countsdataG1.at['RGFr'] != 0 and countsdataR1.at['RGFr'] == 0 and countsdataFr1.at['RGFr'] == 0:
                RGFr = countsdataG1.at['RGFr']
            if countsdataG1.at['RGFr'] == 0 and countsdataR1.at['RGFr'] == 0 and countsdataFr1.at['RGFr'] == 0:
                RGFr = 0
        except:
            RGFr = 0

    if option == 3:
        try:
            if countsdataG1.at['GFr'] == 0 and countsdataFr1.at['GFr'] != 0:
                GFr = countsdataFr1.at['GFr']
            if countsdataFr1.at['GFr'] == 0 and countsdataG1.at['GFr'] != 0:
                GFr = countsdataG1.at['GFr']
            if countsdataG1.at['GFr'] != 0 and countsdataFr1.at['GFr'] != 0:
                GFr = (countsdataG1.at['GFr']+countsdataFr1.at['GFr'])/2
            if countsdataG1.at['GFr'] == 0 and countsdataR1.at['GFr'] == 0:
                GFr = 0
        except:
            GFr = 0
    try:
        R = countsdataR1.at['R']
    except:
        R = 0

    if option == 3:
        try:
            Fr = countsdataFr1.at['Fr']
        except:
            Fr = 0

    if option == 3:
        try:
            if countsdataFr1.at['RFr'] == 0 and countsdataR1.at['RFr'] != 0:
                RFr = countsdataR1.at['RFr']
            if countsdataR1.at['RFr'] == 0 and countsdataFr1.at['RFr'] != 0:
                RFr = countsdataFr1.at['RFr']
            if countsdataFr1.at['RFr'] != 0 and countsdataR1.at['RFr'] != 0:
                RFr = (countsdataFr1.at['RFr']+countsdataR1.at['RFr'])/2
            if countsdataFr1.at['RFr'] == 0 and countsdataR1.at['RFr'] == 0:
                RFr = 0
        except:
            RFr = 0
    Gper = (G/countsdataR1.sum())*100
    RGper = (RG/countsdataR1.sum())*100
    if option == 3:
        RGFrper = (RGFr/countsdataR1.sum())*100
        GFrper = (GFr/countsdataR1.sum())*100
        Frper = (Fr / countsdataR1.sum()) * 100
        RFrper = (RFr / countsdataR1.sum()) * 100
    Rper = (R/countsdataR1.sum())*100


    if option == 3:
        finaldata = [['Red',Rper], ['Red and Green', RGper], ['Red, Green and Far Red',RGFrper], ['Far Red', Frper], ['Green',Gper],['Red and Far Red',RFrper], ['Far Red and Green',GFrper]]
    if option == 2:
        finaldata = [['Red', Rper], ['Red and Green', RGper], ['Green', Gper]]
    df = pd.DataFrame(finaldata, columns=['Sample', 'Percentage'])
    if option == 3:
        return(df,countsdataR1,countsdataG1,countsdataFr1)
    if option == 2:
        return(df,countsdataR1,countsdataG1)

# print(datamaker(2))