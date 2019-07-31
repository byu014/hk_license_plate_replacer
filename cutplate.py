import json, cv2
import numpy as np
import random
import math

def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts, maxWidth=220, maxHeight=70):
    pts = np.array(pts).reshape([4, 2])
    #print(pts)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    dst = np.array([
        [0, 0],
       	[maxWidth - 1, 0],
       	[maxWidth - 1, maxHeight - 1],
       	[0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped, M, maxWidth, maxHeight

def four_point_transform_keep_plate_size(image, pts):
    pts = np.array(pts).reshape([4, 2])
    #print(pts)
    rect = order_points(pts)
    maxHeight = abs(rect[3, 1] - rect[0, 1]) + abs(rect[2, 1] - rect[1, 1])/2
    maxWidth = abs(rect[1, 0] - rect[0, 0]) + abs(rect[2, 0] - rect[3, 0])/2

    (tl, tr, br, bl) = rect
    dst = np.array([
        [0, 0],
       	[maxWidth - 1, 0],
       	[maxWidth - 1, maxHeight - 1],
       	[0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def stitch(source_img, dst_img, corner_pts):
    corner_pts = np.array(corner_pts).reshape([4, 2])
    # source image
    size = source_img.shape
    # get four corners of the source (clock wise)
    pts_source = np.array(
                        [
                        [0,0],
                        [size[1] - 1, 0],
                        [size[1] - 1, size[0] -1],
                        [0, size[0] - 1 ]
                        ],dtype=float
                        )
    #pts_source = np.array([[310,0], [440,0], [589,151],[383,151]])

    # destination image
    # four corners in destination image (also clock wise):
    pts_dst = order_points(corner_pts)

    # calculate homography
    h, status = cv2.findHomography(pts_source, pts_dst)

    # warp source image to destination based on homography
    temp = cv2.warpPerspective(source_img, h, (dst_img.shape[1], dst_img.shape[0]))
    rows, cols, _ = temp.shape

    #random rotation
    angle = random.randint(-2,2)
    centerX , centerY = center_four_points(pts_dst)
    rotation_matrix = cv2.getRotationMatrix2D((centerX,centerY),angle,1,)
    temp = cv2.warpAffine(temp,rotation_matrix,(cols,rows),borderMode=cv2.BORDER_CONSTANT, borderValue = [0, 0, 0, 0])
    pts_dstChanged = pts_shift_angle(pts_dst,angle,centerX,centerY)
    
    #random translation
    randX = random.randint(-5,5)
    randY = random.randint(-5,5)
    M = np.float32([[1,0,randX],[0,1,randY]])
    temp = cv2.warpAffine(temp,M,(cols,rows))
    pts_dstChanged = pts_shift_translation(pts_dstChanged,randX,randY)
    
    #blackspacing license plate areas in dst_img
    cv2.fillConvexPoly(dst_img, pts_dst.astype(int), [0,0,0],16)
    cv2.fillConvexPoly(dst_img, pts_dstChanged.astype(int), [0,0,0],16)

    #lowers brightness of car plate by a miniscule amount
    alpha = .9
    beta = -20
    temp = cv2.addWeighted(temp, alpha, np.zeros(temp.shape, temp.dtype),0, beta)

    #small area masking laplacian 
    mask = np.zeros(dst_img.shape, dtype=dst_img.dtype)
    roi_corners = enlargePoly(pts_dstChanged)
    channel_count = dst_img.shape[2]
    ignore_mask_color = (255,)*channel_count
    cv2.fillConvexPoly(mask, roi_corners.astype(int), ignore_mask_color,16)
    masked_image = cv2.bitwise_and(dst_img, mask)

    #blending synthetic plate with small region of image
    temp = Laplacian_Pyramid_Blending(temp,masked_image)
    temp = cv2.resize(temp,(cols,rows))
    mask = np.zeros(dst_img.shape, dtype=dst_img.dtype)
    cv2.fillConvexPoly(mask, pts_dstChanged.astype(int), ignore_mask_color,16)
    temp = cv2.bitwise_and(temp, mask)
    dst_img += temp

    # alpha = .95
    # beta = -15
    # dst_img = cv2.addWeighted(dst_img, alpha, np.zeros(dst_img.shape, dst_img.dtype),0, beta)
    pts_dstChanged = np.array(pts_dstChanged).reshape([1,8])
    pts_dstChanged = pts_dstChanged.tolist()
    return dst_img , pts_dstChanged

def enlargePoly(pts):
    return_corners = pts.copy()
    
    return_corners[0][0] -= 10
    return_corners[0][1] -= 10
    return_corners[1][0] += 10
    return_corners[1][1] -= 10
    return_corners[2][0] += 10
    return_corners[2][1] += 10
    return_corners[3][0] -= 10
    return_corners[3][1] += 10

    return return_corners

def pts_shift_translation(corner_pts,randX,randY):
    returnPts = np.copy(corner_pts)
    for i,point in enumerate(corner_pts):
        returnPts[i][0] += randX
        returnPts[i][1] += randY
    return returnPts


def pts_shift_angle(corner_pts,angle,tx,ty):
    a = -(angle * math.pi/180)
    returnPts = np.empty_like(corner_pts)
    for i,point in enumerate(corner_pts):
        x = corner_pts[i][0]
        y = corner_pts[i][1]
        # corner_pts[i][0] = y*math.cos(a) - x*math.sin(a)
        # corner_pts[i][1] = y*math.sin(a) + x*math.cos(a)
        returnPts[i][0] = (x-tx)*math.cos(a) - (y-ty)*math.sin(a) + tx
        returnPts[i][1] = (y-ty)*math.cos(a) + (x-tx)*math.sin(a) + ty

    return returnPts.astype(int)

def center_four_points(vertices):
    _x = 0
    _y = 0
    signedArea = 0.0
    x0 = 0.0 # Current vertex X
    y0 = 0.0 # Current vertex Y
    x1 = 0.0 # Next vertex X
    y1 = 0.0 # Next vertex Y
    a = 0.0  # Partial signed area

    # For all vertices
    for i in range(len(vertices)):
        x0 = vertices[i][0]
        y0 = vertices[i][1]
        x1 = vertices[(i+1) % len(vertices)][0]
        y1 = vertices[(i+1) % len(vertices)][1]
        a = x0*y1 - x1*y0
        signedArea += a
        _x += (x0 + x1)*a
        _y += (y0 + y1)*a

    signedArea *= 0.5
    _x /= (6.0*signedArea)
    _y /= (6.0*signedArea)

    return _x,_y

def Laplacian_Pyramid_Blending(A,B):
   # generate Gaussian pyramid for A
    num = 3
    G = A.copy()
    gpA = [G]
    for i in range(num):
        G = cv2.pyrDown(G)
        gpA.append(G)

    # generate Gaussian pyramid for B
    G = B.copy()
    gpB = [G]
    for i in range(num):
        G = cv2.pyrDown(G)
        gpB.append(G)

    lpA = [gpA[num-1]]
    for i in range(num,0,-1):
        GE = cv2.pyrUp(gpA[i])
        GE=cv2.resize(GE,gpA[i - 1].shape[-2::-1])
        L = cv2.subtract(gpA[i-1],GE)
        lpA.append(L)

    # generate Laplacian Pyramid for B

    lpB = [gpB[num-1]]
    for i in range(num,0,-1):
        GE = cv2.pyrUp(gpB[i])
        GE = cv2.resize(GE, gpB[i - 1].shape[-2::-1])
        L = cv2.subtract(gpB[i-1],GE)
        lpB.append(L)

    # Now add left and right halves of images in each level
    LS = []
    lpAc=[]
    for i in range(len(lpA)):
        b=cv2.resize(lpA[i],lpB[i].shape[-2::-1])
        lpAc.append(b)
    j=0
    for i in zip(lpAc,lpB):
        la,lb = i
        rows,cols,dpt = la.shape
        # ls = np.hstack((la[:,:], lb[:,:]))
        ls = la + lb
        j=j+1
        LS.append(ls)

    ls_ = LS[0]
    for i in range(1,num):
        ls_ = cv2.pyrUp(ls_)
        ls_= cv2.resize(ls_, LS[i].shape[-2::-1])
        ls_ = cv2.add(ls_, LS[i])
    return ls_

if __name__ == '__main__':
    img = cv2.imread('./ch12_20180805095959.mp4_016125_box_1.jpg')
    j = json.load(open('./ch12_20180805095959.mp4_016125_box_1.jpg.out.json'))['text_lines'][0]
    points = np.array([[j['x0'], j['y0']], [j['x1'], j['y1']],
              [j['x2'], j['y2']], [j['x3'], j['y3']]])
    print(points)
    img = four_point_transform(img, points)
    cv2.imshow('a', img)
    cv2.waitKey()
