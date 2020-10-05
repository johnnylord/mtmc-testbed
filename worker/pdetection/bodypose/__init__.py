import logging
import math
import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import torch

from ...utils.time import timeit
from ..base import PoseDetector
from .model import bodypose_model
from .util import transfer

logger = logging.getLogger(__name__)


class BodyPoseDetector(PoseDetector):
    """Openpose bodypose estimation model trained on COCO dataset

    Reference:
        https://arxiv.org/abs/1611.08050
    """
    PRETRAIN_URL = "https://www.dropbox.com/sh/7xbup2qsn7vvjxo/AABaYNMvvNVFRWqyDXl7KQUxa/body_pose_model.pth?dl=1"

    POSE_NAMES = [ "Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder",
                "LElbow", "LWrist", "MidHip", "RHip", "RKnee", "RAnkle", "LHip",
                "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar" ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load pretrain state
        if self.pretrain_model is None:
            state_dict = torch.hub.load_state_dict_from_url(BodyPoseDetector.PRETRAIN_URL)
        else:
            state_dict = torch.loads(self.PRETRAIN_MODEL)

        # Instantiate model
        model = bodypose_model()
        state_dict = transfer(model, state_dict)
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        self.model = model.eval()

        # Construct mapping table
        self.pose2id = dict([ (n, i) for i, n in enumerate(BodyPoseDetector.POSE_NAMES) ])
        self.id2pose = dict([ (i, n) for i, n in enumerate(BodyPoseDetector.POSE_NAMES) ])

    def preprocessing(self, imgs):
        # Convert list to numpy array (n_imgs, img_height, img_width, channels)
        imgs = np.array(imgs)

        # Transform imgs
        imgs = np.transpose(np.float32(imgs), (0, 3, 1, 2))
        imgs = imgs / 256 - 0.5
        imgs = np.ascontiguousarray(imgs)

        # Transform to tensor
        data = torch.from_numpy(imgs).float()
        data = data.to(self.device)

        return data

    @timeit(logger)
    def postprocessing(self, imgs, heatmaps, pafs):
        # All img metadata
        img_width = imgs[0].shape[1]
        img_height = imgs[0].shape[0]
        img_channel = imgs[0].shape[2]

        # Recover heatmaps and pafs to original img coordinate system
        rec_heatmaps = []
        rec_pafs = []

        heatmaps = heatmaps.detach().cpu().numpy()
        pafs = pafs.detach().cpu().numpy()
        for idx, (heatmap, paf) in enumerate(zip(heatmaps, pafs)):
            heatmap = np.transpose(heatmap, (1, 2, 0))
            heatmap = cv2.resize(heatmap,
                                (img_width, img_height),
                                interpolation=cv2.INTER_CUBIC)
            paf = np.transpose(paf, (1, 2, 0))
            paf = cv2.resize(paf,
                            (img_width, img_height),
                            interpolation=cv2.INTER_CUBIC)
            rec_heatmaps.append(heatmap)
            rec_pafs.append(paf)

        heatmaps = np.array(rec_heatmaps)
        pafs = np.array(rec_pafs)

        # Greedy Matching
        # ====================================================================
        peoples = []
        for heatmap, paf in zip(heatmaps, pafs):
            # Find all kinds of peaks in heatmap of shape (height, width, 18)
            # ================================================================
            all_peaks = []
            peak_counter = 0
            for part in range(18):
                map_ori = heatmap[:, :, part]
                one_heatmap = gaussian_filter(map_ori, sigma=3)

                # Create a cross ('+') heatmap as peaks are found with
                # gaussian probability distributed function in mind
                map_left = np.zeros(one_heatmap.shape)
                map_left[1:, :] = one_heatmap[:-1, :]
                map_right = np.zeros(one_heatmap.shape)
                map_right[:-1, :] = one_heatmap[1:, :]
                map_up = np.zeros(one_heatmap.shape)
                map_up[:, 1:] = one_heatmap[:, :-1]
                map_down = np.zeros(one_heatmap.shape)
                map_down[:, :-1] = one_heatmap[:, 1:]
                peaks_binary = np.logical_and.reduce((
                                                    one_heatmap >= map_left,
                                                    one_heatmap >= map_right,
                                                    one_heatmap >= map_up,
                                                    one_heatmap >= map_down,
                                                    one_heatmap > self.hthreshold))
                # Find (x, y) position of all peaks
                peaks = list(zip(np.nonzero(peaks_binary)[1], # x
                                np.nonzero(peaks_binary)[0])) # y
                # Find (x ,y, score) of all peaks
                peaks_with_score = [x + (map_ori[x[1], x[0]],)
                                    for x in peaks]
                # Find (x, y, score, id) of all peaks
                peak_id = range(peak_counter, peak_counter + len(peaks))
                peaks_with_score_and_id = [ peaks_with_score[i] + (peak_id[i],)
                                            for i in range(len(peak_id))]

                all_peaks.append(peaks_with_score_and_id)
                peak_counter += len(peaks)

            # Find limbs to connect peaks
            # =================================================================
            # Find connection in the specified sequence
            # NOTE: the number represents the index of heatmap
            limbSeq = [ [2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9],\
                        [9, 10], [10, 11], [2, 12], [12, 13], [13, 14], [2, 1],\
                        [1, 15], [15, 17], [1, 16], [16, 18], [3, 17], [6, 18]]
            # Corresponding index of PAF map for each limb
            #   - shape of paf is (height, width, 36)
            #   - paf estimated the orientation of each pair of limb. Therefore,
            #   each pair of limb has two paf maps: (0, 1), (2, 3), and etc.
            # NOTE: the number represents the index of paf map
            mapIdx = [  [31, 32], [39, 40], [33, 34], [35, 36], [41, 42], \
                        [43, 44], [19, 20], [21, 22], [23, 24], [25, 26], \
                        [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], \
                        [51, 52], [55, 56], [37, 38], [45, 46]]

            mid_num = 10
            special_k = []
            connection_all = []
            for k in range(len(mapIdx)):
                # Prepare limb peaks and associated paf maps
                candA = all_peaks[limbSeq[k][0] - 1]
                candB = all_peaks[limbSeq[k][1] - 1]
                score_mid = paf[:, :, [x - 19 for x in mapIdx[k]]]

                # Find matching between candA and candB (find possible limbs)
                nA = len(candA)
                nB = len(candB)
                indexA, indexB = limbSeq[k]
                if (nA != 0 and nB != 0):
                    connection_candidate = []

                    # All combination between candA and candB
                    for i in range(nA):
                        for j in range(nB):
                            # Find unit vector pointing from candA[i] to candB[j]
                            vec = np.subtract(candB[j][:2], candA[i][:2])
                            norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                            vec = np.divide(vec, norm)

                            # Intermediate points between candA[i], candB[j]
                            A_x, B_x = candA[i][0], candB[j][0]
                            A_y, B_y = candA[i][1], candB[j][1]
                            startend = list(zip(np.linspace(A_x, B_x, mid_num),
                                                np.linspace(A_y, B_y, mid_num)))

                            # Orientation vectors of intermediate points
                            vec_x = np.array([score_mid[int(round(startend[i][1])),
                                                        int(round(startend[i][0])),
                                                        0]
                                              for i in range(len(startend))])
                            vec_y = np.array([score_mid[int(round(startend[i][1])),
                                                        int(round(startend[i][0])),
                                                        1]
                                              for i in range(len(startend))])

                            # Caculate the integral value between candA[i] & candB[j]
                            score_midpts_x = np.multiply(vec_x, vec[0])
                            score_midpts_y = np.multiply(vec_y, vec[1])
                            score_midpts = score_midpts_x + score_midpts_y

                            # Filter out invalid pair (candA[i], candB[j])
                            mean = sum(score_midpts)/len(score_midpts)
                            prior = min(0.5*img_height/(norm+1e-6)-1, 0)
                            score_with_dist_prior = mean + prior
                            n_valids = len(np.nonzero(score_midpts > self.pthreshold)[0])
                            if (
                                score_with_dist_prior > 0
                                and n_valids > 0.8 * len(score_midpts)
                            ):
                                connection_candidate.append(
                                    [i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]])

                    # connection's shape (n, 5)
                    #   - axis 1 represents (A_id, B_id, score, A_idx, B_idx)
                    connection = np.zeros((0, 5))
                    connection_candidate = sorted(connection_candidate,
                                                key=lambda x: x[2], reverse=True)
                    for c in range(len(connection_candidate)):
                        i, j, s = connection_candidate[c][0:3]
                        # One-to-one connection constraint
                        if (
                            i not in connection[:, 3]
                            and j not in connection[:, 4]
                        ):
                            connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                            if (len(connection) >= min(nA, nB)):
                                break

                    connection_all.append(connection)
                else:
                    special_k.append(k)
                    connection_all.append([])

            # Group limbs together to form multiple groups
            #   - subset means a group of people
            #   - candidate means an array of keypoints
            # ================================================================
            subset = -1 * np.ones((0, 20))
            candidate = np.array([  item
                                    for sublist in all_peaks
                                    for item in sublist ])
            for k in range(len(mapIdx)):
                if k not in special_k:
                    # Candidate ids
                    A_ids = connection_all[k][:, 0]
                    B_ids = connection_all[k][:, 1]
                    # Limb indice
                    A_idx, B_idx = np.array(limbSeq[k]) - 1

                    for i in range(len(connection_all[k])):
                        # Check whether connection ends in existing people
                        found = 0
                        subset_idx = [-1, -1]
                        for j in range(len(subset)):
                            if (
                                subset[j][A_idx] == A_ids[i]
                                or subset[j][B_idx] == B_ids[i]
                            ):
                                subset_idx[found] = j
                                found += 1

                        # Merge to existing person
                        if found == 1:
                            j = subset_idx[0]
                            if subset[j][B_idx] != B_ids[i]:
                                subset[j][B_idx] = B_ids[i]
                                subset[j][-1] += 1
                                subset[j][-2] += candidate[B_ids[i].astype(int), 2]
                                subset[j][-2] += connection_all[k][i][2]

                        # Merge two people together
                        elif found == 2:
                            j1, j2 = subset_idx
                            n1_points = (subset[j1] >= 0).astype(int)
                            n2_points = (subset[j2] >= 0).astype(int)
                            membership = (n1_points+n2_points)[:-2]

                            # Merge if two people are disjoint set
                            if len(np.nonzero(membership == 2)[0]) == 0:
                                subset[j1][:-2] += (subset[j2][:-2] + 1)
                                subset[j1][-2:] += subset[j2][-2:]
                                subset[j1][-2] += connection_all[k][i][2]
                                subset = np.delete(subset, j2, 0)

                            # Merge to existing person
                            else:
                                subset[j1][B_idx] = B_ids[i]
                                subset[j1][-1] += 1
                                subset[j1][-2] += candidate[B_ids[i].astype(int), 2]
                                subset[j1][-2] += connection_all[k][i][2]

                        # Create a new person
                        elif not found and k < 17:
                            row = -1 * np.ones(20)
                            row[A_idx] = A_ids[i]
                            row[B_idx] = B_ids[i]
                            row[-1] = 2
                            row[-2] = connection_all[k][i][2]
                            row[-2] += sum(candidate[connection_all[k][i, :2].astype(int), 2])
                            subset = np.vstack([subset, row])

            # delete some rows of subset which has few parts occur
            indices = [ i for i in range(len(subset))
                        if (subset[i][-1] < 4
                            or subset[i][-2] / subset[i][-1] < 0.4) ]
            subset = np.delete(subset, indices, axis=0)

            # Construct keypoints of each person
            people = []
            for person in subset:
                # Keypoints part
                score = person[-2]
                n_parts = person[-1]
                keypoints = np.zeros((18, 3))
                for idx, cid in enumerate(person[:18]):
                    if cid == -1:
                        continue
                    keypoints[idx] = candidate[int(cid)][:3]

                # Bbox part (xmin, ymin, xmax, ymax)
                xmin, ymin = 10000, 10000
                xmax, ymax = -1, -1
                for keypoint in keypoints:
                    if keypoint[-1] == 0:
                        continue
                    kx, ky = keypoint[0], keypoint[1]
                    if kx > xmax:
                        xmax = kx
                    if kx < xmin:
                        xmin = kx
                    if ky > ymax:
                        ymax = ky
                    if ky < ymin:
                        ymin = ky
                bbox = (xmin, ymin, xmax, ymax)

                # Append person
                people.append({ 'conf': score,
                                'bbox': bbox,
                                'n_parts': n_parts,
                                'keypoints': keypoints })
            peoples.append(people)

        return peoples
