import numpy as np
import cv2 as cv2
from numpy.linalg import inv, pinv
import matplotlib.pyplot as plt
import time as t

def getK():
    return np.array([[7.188560e+02, 0.000000e+00, 6.071928e+02],
                     [0, 7.188560e+02, 1.852157e+02],
                     [0, 0, 1]])

def getTruePose():
    file = 'E:/DataSets/dataset/2011_10_03_drive_0027_sync/2011_10_03/2011_10_03_drive_0027_sync/poses/00.txt'
    return np.genfromtxt(file, delimiter=' ', dtype=None)

def getLeftImage(i):
    return cv2.imread('E:/DataSets/dataset/2011_10_03_drive_0027_sync/2011_10_03/2011_10_03_drive_0027_sync/image_00/data/{0:010d}.png'.format(i), 0)

def getRightImage(i):
    return cv2.imread('E:/DataSets/dataset/2011_10_03_drive_0027_sync/2011_10_03/2011_10_03_drive_0027_sync/image_01/data/{0:010d}.png'.format(i), 0)

def extract_keypoints_surf(img1, img2, K, baseline):

    ##### ##################################### #######
    ##### Get Feature detection and Description #######
    ##### #############################################
    detector = cv2.xfeatures2d.SURF_create(400)
    kp1, desc1 = detector.detectAndCompute(img1, None)
    kp2, desc2 = detector.detectAndCompute(img2, None)

    ##### ############# #######
    ##### Match Points #######
    ##### ####################
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict()  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)

    # ratio test as per Lowe's paper
    match_points1, match_points2 = [], []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            match_points1.append(kp1[m.queryIdx].pt)
            match_points2.append(kp2[m.trainIdx].pt)

    p1 = np.array(match_points1).astype(float)
    p2 = np.array(match_points2).astype(float)

    ##### ############# ##########
    ##### Do Triangulation #######
    ##### ########################
    #projection matrix for Left and Right Image
    M_left = K.dot(np.hstack((np.eye(3), np.zeros((3, 1)))))
    M_rght = K.dot(np.hstack((np.eye(3), np.array([[-baseline, 0, 0]]).T)))

    p1_flip = np.vstack((p1.T, np.ones((1, p1.shape[0]))))
    p2_flip = np.vstack((p2.T, np.ones((1, p2.shape[0]))))

    P = cv2.triangulatePoints(M_left, M_rght, p1_flip[:2], p2_flip[:2])

    #Normalize homogeneous coordinates (P->Nx4  [N,4] is the normalizer/scale)
    P = P / P[3]
    land_points = P[:3]

    return land_points.T, p1

def featureTracking(img_1, img_2, p1, world_points):
    ##use KLT tracker
    lk_params = dict(winSize=(21, 21),
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    p2, st, err = cv2.calcOpticalFlowPyrLK(img_1, img_2, p1, None, **lk_params)
    st = st.reshape(st.shape[0])
    ##find good ones
    pre = p1[st == 1]
    p2 = p2[st == 1]
    w_points = world_points[st == 1]

    return w_points, pre, p2

def playImageSequence(left_img, right_img, K):

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('project_orig.avi', fourcc, 30, (1841, 600))
    ##### ################################# #######
    ##### Get 3D points Using Triangulation #######
    ##### #########################################
    points, p1 = extract_keypoints_surf(left_img, right_img, K, baseline)
    p1 = p1.astype('float32')

    # reference
    reference_img = left_img
    reference_2D = p1
    landmark_3D = points

    # Groundtruth for plot
    truePose = getTruePose()
    traj = np.zeros((600, 600, 3), dtype=np.uint8);
    maxError = 0

    for i in range(0, 1400):
        print('image: ', i)
        curImage = getLeftImage(i)
        # curImage = cv2.imread('../data/left/{0:06d}.png'.format(i), 0)


        ##### ############################################################# #######
        ##### Calculate 2D and 3D feature correspndances in t=T-1, and t=T  #######
        ##### #####################################################################
        landmark_3D, reference_2D, tracked_2Dpoints = featureTracking(reference_img, curImage, reference_2D,
                                                                      landmark_3D)

        ##### ################################# #######
        ##### Calculate relative pose using PNP #######
        ##### #########################################
        pnp_objP = np.expand_dims(landmark_3D, axis=2)
        pnp_cur = np.expand_dims(tracked_2Dpoints, axis=2).astype(float)
        _, rvec, tvec, inliers = cv2.solvePnPRansac(pnp_objP, pnp_cur, K, None)

        ##### ####################################################### #######
        ##### Get Pose and Tranformation Matrix in world coordionates #######
        ##### ###############################################################
        rot, _ = cv2.Rodrigues(rvec)
        tvec = -rot.T.dot(tvec)  # coordinate transformation, from camera to world. What is the XYZ of the camera wrt World
        inv_transform = np.hstack((rot.T, tvec))  # inverse transform. A tranform projecting points from the camera frame to the world frame

        ##### ################################# #######
        ##### Get 3D points Using Triangulation #######
        ##### #########################################
        # re-obtain the 3 D points
        curImage_R = getRightImage(i)
        landmark_3D_new, reference_2D_new = extract_keypoints_surf(curImage, curImage_R, K, baseline )
        reference_2D = reference_2D_new.astype('float32')
        #Project the points from camera to world coordinates
        landmark_3D = inv_transform.dot(np.vstack((landmark_3D_new.T, np.ones((1, landmark_3D_new.shape[0])))))
        landmark_3D = landmark_3D.T

        ##### ####################### #######
        ##### Done, Next image please #######
        ##### ###############################
        reference_img = curImage

        ##### ################################## #######
        ##### START OF Print and visualize stuff #######
        ##### ##########################################
        # draw images
        draw_x, draw_y = int(tvec[0]) + 300, 600-(int(tvec[2]) + 100);
        true_x, true_y = int(truePose[i][3]) + 300, 600-(int(truePose[i][11]) + 100)

        curError = np.sqrt(
            (tvec[0] - truePose[i][3]) ** 2 + (tvec[1] - truePose[i][7]) ** 2 + (tvec[2] - truePose[i][11]) ** 2)
        # print('Current Error: ', curError)
        if (curError > maxError):
            maxError = curError

        print(tvec[0],tvec[1],tvec[2], rvec[0], rvec[1], rvec[2])
        print([truePose[i][3], truePose[i][7], truePose[i][11]])
        # print([truePose[i][3], truePose[i][7], truePose[i][11]])
        text = "Coordinates: x ={0:02f}m y = {1:02f}m z = {2:02f}m".format(float(tvec[0]), float(tvec[1]),
                                                                           float(tvec[2]));
        cv2.circle(traj, (draw_x, draw_y), 1, (0, 0, 255), 2);
        cv2.circle(traj, (true_x, true_y), 1, (255, 0, 0), 2);
        cv2.rectangle(traj, (10, 30), (550, 50), (0, 0, 0), cv2.FILLED);
        cv2.putText(traj, text, (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8);

        h1, w1 = traj.shape[:2]
        h2, w2 = curImage.shape[:2]
        vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
        vis[:h1, :w1, :3] = traj
        vis[:h2, w1:w1 + w2, :3] = np.dstack((np.dstack((curImage,curImage)),curImage))

        cv2.imshow("Trajectory", vis);
        out.write(vis)
        k = cv2.waitKey(1) & 0xFF
        if k == 27: break

    out.release()
    cv2.waitKey(0)
    print('Maximum Error: ', maxError)
    ##### ################################ #######
    ##### END OF Print and visualize stuff #######
    ##### ########################################

if __name__ == '__main__':
    left_img = getLeftImage(0)
    right_img = getRightImage(0)

    baseline = 0.54;
    K = getK()

    playImageSequence(left_img, right_img, K)
