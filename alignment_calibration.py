import czifile
import numpy as np
import cv2
import tifffile as tiff
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
startTime = datetime.now()

def getmatrix(controlimage):
    inputfile = controlimage
    img = czifile.imread(inputfile)
    img1 = img[0, :, :, :, 0]
    reds = img1[0, :, :]
    greens = img1[1, :, :]
    blues = img1[2, :, :]
    farreds = img1[3, :, :]

    ratior = np.amax(reds) / 256
    r = (reds / ratior).astype('uint8')
    ratiog = np.amax(greens) / 256
    g = (greens / ratiog).astype('uint8')
    ratiob = np.amax(blues) / 256
    b = (blues / ratiob).astype('uint8')
    ratiofr = np.amax(farreds) / 256
    fr = (farreds / ratiofr).astype('uint8')

    sz = r.shape
    warp_mode = cv2.MOTION_HOMOGRAPHY
    warp_matrix = np.eye(3, 3, dtype=np.float32)

    number_of_iterations = 5000

    termination_eps = 1e-10

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    cc, warp_matrix1 = cv2.findTransformECC(g, b, warp_matrix, warp_mode, criteria)
    print("[alignment 1 done]")
    blues_aligned = cv2.warpPerspective(blues, warp_matrix1, (sz[1], sz[0]),
                                        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    cc2, warp_matrix2 = cv2.findTransformECC(g, r, warp_matrix, warp_mode, criteria)
    print("[alignment 2 done]")
    reds_aligned = cv2.warpPerspective(reds, warp_matrix2, (sz[1], sz[0]),
                                         flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    ratiorbl = np.amax(reds) / 256
    ral = (reds_aligned / ratiorbl).astype('uint8')

    cc3, warp_matrix3 = cv2.findTransformECC(ral, fr, warp_matrix, warp_mode, criteria)

    farreds_aligned = cv2.warpPerspective(farreds, warp_matrix3, (sz[1], sz[0]),
                                         flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    print("[alignment 3 done]")
    # Show final results
    before = np.zeros((blues.shape[0], blues.shape[1], 4), dtype=np.uint16)
    before[:, :, 0] = reds
    before[:, :, 1] = greens
    before[:, :, 2] = blues
    before[:, :, 3] = farreds
    tiff.imsave('control_before.tiff', before)

    after = np.zeros((blues.shape[0], blues.shape[1], 4), dtype=np.uint16)
    after[:, :, 0] = reds_aligned
    after[:, :, 1] = greens
    after[:, :, 2] = blues_aligned
    after[:, :, 3] = farreds_aligned
    tiff.imsave('control_after.tiff', after)
    # warp_matrix3="dave"
    return(warp_matrix1,warp_matrix2, warp_matrix3)

def imagealign(image,warp1,warp2,warp3):
    inputfile = image
    img = czifile.imread(inputfile)
    img1 = img[0, :, :, :, 0]
    reds = img1[0, :, :]
    greens = img1[1, :, :]
    blues = img1[2, :, :]
    farreds = img1[3, :, :]
    sz = reds.shape

    blues_aligned = cv2.warpPerspective(blues, warp1, (sz[1], sz[0]),flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    reds_aligned = cv2.warpPerspective(reds, warp2, (sz[1], sz[0]),flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    farreds_aligned = cv2.warpPerspective(farreds, warp3, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    filename=inputfile[:-4]

    before = np.zeros((blues.shape[0], blues.shape[1], 4), dtype=np.uint16)
    before[:, :, 0] = reds
    before[:, :, 1] = greens
    before[:, :, 2] = blues
    before[:, :, 3] = farreds
    beforename=filename+"_before.tiff"
    tiff.imsave(beforename, before)

    after = np.zeros((blues.shape[0], blues.shape[1], 4), dtype=np.uint16)
    after[:, :, 0] = reds_aligned
    after[:, :, 1] = greens
    after[:, :, 2] = blues_aligned
    after[:, :, 3] = farreds_aligned
    aftername=filename+"_after.tiff"
    tiff.imsave(aftername, after)

def runit(file):
    warp1,warp2,warp3=getmatrix(file)
    print(datetime.now() - startTime)

# print(runit('./RawDataFromScottie/2hr/211213AHGM28LE.czi'))

# runit("./RawDataFromScottie/211009AHGMT28H10.czi")
# imagealign("28hibeads10.czi",warp1,warp2,warp3)

#
# inputfile='driedbeads.czi'
# img = czifile.imread(inputfile)
# img1=img[0, :, :, :, 0]
# reds=img1[0,:,:]
# greens=img1[1,:,:]
# blues=img1[2,:,:]
#
# ratior = np.amax(reds) / 256
# r = (reds / ratior).astype('uint8')
# ratiog = np.amax(greens) / 256
# g = (greens / ratiog).astype('uint8')
# ratiob = np.amax(blues) / 256
# b = (blues / ratiob).astype('uint8')
# #
# # r = np.uint8(reds)
# # g = np.uint8(greens)
# # b = np.uint8(blues)
#
#
# before = np.zeros((b.shape[0], b.shape[1], 3),dtype=np.uint8)
# before [:,:,0] = r
# before [:,:,1] = g
# before [:,:,2] = b
# tiff.imsave('before.tiff', before)
#
# sz = r.shape
# warp_mode = cv2.MOTION_HOMOGRAPHY
# warp_matrix = np.eye(3, 3, dtype=np.float32)
#
# number_of_iterations = 100
#
# termination_eps = 1e-10
#
# # Define termination criteria
# criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
#
# cc, warp_matrix = cv2.findTransformECC(r,b,warp_matrix, warp_mode, criteria)
#
# blues_aligned = cv2.warpPerspective(blues, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
#
# cc2, warp_matrix2 = cv2.findTransformECC(r,g,warp_matrix, warp_mode, criteria)
#
# greens_aligned = cv2.warpPerspective(greens, warp_matrix2, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
#
# # Show final results
# before = np.zeros((blues.shape[0], blues.shape[1], 3),dtype=np.uint16)
# before [:,:,0] = reds
# before [:,:,1] = greens
# before [:,:,2] = blues
# tiff.imsave('before.tiff', before)
#
# after = np.zeros((blues.shape[0], blues.shape[1], 3),dtype=np.uint16)
# after [:,:,0] = reds
# after [:,:,1] = greens_aligned
# after [:,:,2] = blues_aligned
# tiff.imsave('after.tiff', after)
#
#
#
#
#
#



