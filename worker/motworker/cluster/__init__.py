import logging

import numpy as np
from scipy.optimize import linear_sum_assignment

from .centroid import TargetCentroid


logger = logging.getLogger(__name__)


class AdaptiveKmeans:
    """Adaptive Kmeans clustering algorithm to cluster tracked targets"""
    NEXT_TID = 0

    def __init__(self):
        self.centroids = {}

    def __str__(self):
        lines = [ "[TID:{}:{}]".format(tid, tc) for tid, tc in self.centroids.items() ]
        content = ", ".join(lines)
        return content

    def __repr__(self):
        return str(self)

    def miss(self):
        """Perform miss action on all tracked centroids"""
        # Miss for all centroids
        _ = [ tc.miss() for tid, tc in self.centroids.values() ]

        # Delete dead centroids
        tids = list(self.centroids.keys())
        for tid in tids:
            if self.centroids[tid].is_dead():
                del self.centroids[tid]

    def predict(self, points):
        """Predict the track ID for each data point

        Arguments:
            points (ndarray): 2D ndarray representing embeddings from one video

        Returns:
            a list of track ids representing the label of each points

        NOTE:
            `points` should only contain the embeddings from one video. As the
            labels(track IDs) are not determined by the minimum distance between
            points and clusters. They are determined by the result of linear
            assignment algorithm. Each unique label (track ID) will only
            associate with one point.

            The number of centroids should always larger than the number of
            points.
        """
        # Common setup
        centroids = np.array([ tc.embedding for tc in self.centroids.values() ])
        label2tid = dict([ (label, tid)
                        for label, tid in enumerate(self.centroids.keys()) ])

        # Predict tids for points
        distances = self._pdist(points, centroids)
        pindices, cindices = linear_sum_assignment(distances)
        tids = np.array([ label2tid[cidx] for cidx in cindices ])

        return tids

    def fit(self, group_points, n_clusters):
        """Perform adaptive kmeans clustering

        Arguments:
            group_points (list): list of ndarrays, where each element in the
                list representing the embeddings of targets in specific frame.
            n_clusters (int): the ideal number of clusters in current state
        """
        # Flatten group_points
        points = np.concatenate(group_points)

        # Initialize clusters
        if len(self.centroids) == 0:
            self._init_centroids(points, n_clusters)
            return

        # Common setup
        centroids = np.array([ tc.embedding for tc in self.centroids.values() ])
        label2tid = dict([ (label, tid)
                        for label, tid in enumerate(self.centroids.keys()) ])

        # Dynamic add new clusters
        if len(self.centroids) < n_clusters:
            logger.info("ADD NEW CLUSTERS {}".format(n_clusters))
            # Extract anomaly points group by group to form two sets:
            #   - normal points
            #   - anomaly points
            normal_points, anomaly_points = [], []
            for gpoints in group_points:
                # Perform assignment to find best match for each point
                distances = self._pdist(gpoints, centroids)
                sorted_labels = np.argsort(distances)

                # As point to centroid is a one-to-one mapping in each group,
                # filter out points that get assigned to the centroids that
                # already assigned to some points before
                unique_cindices = set()
                for pidx, cindices in enumerate(sorted_labels):
                    cidx = cindices[0]
                    if cidx in unique_cindices:
                        anomaly_points.append(gpoints[pidx])
                    else:
                        normal_points.append(gpoints[pidx])
                    unique_cindices.add(cidx)

            anomaly_points = np.array(anomaly_points)
            normal_points = np.array(normal_points)

            # Add new clusters to fit anomaly points
            new_clusters = n_clusters - len(self.centroids)
            logger.info(anomaly_points)
            self._init_centroids(anomaly_points, new_clusters)

            # Normal points for updating current clusters
            points = normal_points

        # Update existing clusters
        distances = self._pdist(points, centroids)
        labels = np.argmin(distances, axis=1)

        # Compute new centroids
        new_centroids = np.array([  points[labels==label].mean(axis=0)
                                    if np.sum(labels==label) > 0 else c
                                    for label, c in enumerate(centroids) ])

        # Replace new clusters
        for label, c in enumerate(new_centroids):
            tid = label2tid[label]
            self.centroids[tid].embedding = c

        # Update state of clusters
        hit_labels = list(set(labels))
        miss_labels = list(set(range(len(centroids)))-set(labels))
        for hlabel in hit_labels:
            tid = label2tid[hlabel]
            self.centroids[tid].hit()
        for mlabel in miss_labels:
            tid = label2tid[mlabel]
            self.centroids[tid].miss()

        # Cleanup outdated clusters
        tids = list(self.centroids.keys())
        for tid in tids:
            if self.centroids[tid].is_dead():
                del self.centroids[tid]

        # Merge clusters that are too close to each other
        centroids = np.array([ tc.embedding for tc in self.centroids.values() ])
        label2tid = dict([ (label, tid)
                        for label, tid in enumerate(self.centroids.keys()) ])

        unique_clusters = []
        distances = self._pdist(centroids, centroids)
        logger.info("Centroids similarity:\n"+str(distances))

        for cidx, distance in enumerate(distances):
            cluster = set(np.argwhere(distance < 0.4).reshape(-1).tolist())

            merge_flag = False
            for i in range(len(unique_clusters)):
                unique_cluster = unique_clusters[i]
                if len(unique_cluster.intersection(cluster)) > 0:
                    unique_clusters[i] = unique_cluster.union(cluster)
                    merge_flag = True
                    break

            if not merge_flag:
                unique_clusters.append(cluster)

        # Merge clusters
        for clusters in unique_clusters:
            if len(clusters) == 1:
                continue
            tids = sorted([ label2tid[cidx] for cidx in clusters ])
            embeddings = np.array([ self.centroids[tid].embedding
                                    for tid in tids ])
            new_centroid = np.mean(embeddings, axis=0)
            self.centroids[tids[0]].embedding = new_centroid
            for tid in tids[1:]:
                del self.centroids[tid]

        logger.info(str(unique_clusters))

    def _init_centroids(self, points, n_clusters):
        """Initialize clusters that fit the specified points

        Arguments:
            points (ndarray): 2D ndarray data for clustering
            n_clusters (int): number of clusters to initialize
        """
        # Random select centroids from current data points
        centroids = points.copy()
        np.random.shuffle(centroids)
        centroids = centroids[:n_clusters]

        # Fine-tune centroids that best fit data points
        centroids = self._fit(points, centroids)
        for c in centroids:
            self.centroids[AdaptiveKmeans.NEXT_TID] = TargetCentroid(embedding=c)
            AdaptiveKmeans.NEXT_TID += 1

    def _pdist(self, points, centroids):
        """Compute pair-wise distance between data points and centroids

        Arguments:
            points (ndarray): 2D ndarray representing data points with N rows
            centroids (ndarray): 2D ndarray representing centroids with M rows

        Returns:
            A NxM 2D ndarray representing the euclidean distances between data
            points and centroids
        """
        dists = np.sqrt(((points[:, np.newaxis, :]-centroids)**2).sum(axis=2))
        return dists

    def _fit(self, points, centroids, n_iter=10, threshold=1e-3):
        """Perform kmeans algorithm to fit the centroids to the data points

        Arguments:
            points (ndarray): 2D ndarray representing data points
            centroids (ndarray): 2D ndarray representing centroids

        Returns:
            A 2D ndarray representing the fine-tuned centroids
        """
        counter = 0
        while counter < n_iter:
            # Find closet centroid to each point
            distances = self._pdist(points, centroids)
            labels = np.argmin(distances, axis=1)

            # Compute new centroids
            new_centroids = np.array([  points[labels==label].mean(axis=0)
                                        if np.sum(labels==label) > 0 else c
                                        for label, c in enumerate(centroids) ])

            # Break when converge
            diff = np.sum(np.sqrt(((centroids - new_centroids)**2).sum(axis=1)))
            if diff > threshold:
                centroids = new_centroids
            else:
                break

            counter += 1

        return new_centroids
