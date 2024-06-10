import numpy as np
from prettytable import PrettyTable


class Metric_RayPQ:
    def __init__(self, 
                 num_classes=18,
                 thresholds=[1, 2, 4]):
        """
        Args:
            ignore_index (llist): Class ids that not be considered in pq counting.
        """
        if num_classes == 18:
            self.class_names = [
                'others','barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
                'driveable_surface', 'other_flat', 'sidewalk',
                'terrain', 'manmade', 'vegetation','free'
            ]
        else:
            raise ValueError
        
        self.num_classes = num_classes
        self.id_offset = 2 ** 16
        self.eps = 1e-5
        self.thresholds = thresholds
        
        self.min_num_points = 10
        self.include = np.array(
            [n for n in range(self.num_classes - 1)],
            dtype=int)
        self.cnt = 0
        
        # panoptic stuff
        self.pan_tp = np.zeros([len(self.thresholds), num_classes], dtype=int)
        self.pan_iou = np.zeros([len(self.thresholds), num_classes], dtype=np.double)
        self.pan_fp = np.zeros([len(self.thresholds), num_classes], dtype=int)
        self.pan_fn = np.zeros([len(self.thresholds), num_classes], dtype=int)
        
    def add_batch(self,semantics_pred,semantics_gt,instances_pred,instances_gt, l1_error):
        self.cnt += 1
        self.add_panoptic_sample(semantics_pred, semantics_gt, instances_pred, instances_gt, l1_error) 
    
    def add_panoptic_sample(self, semantics_pred, semantics_gt, instances_pred, instances_gt, l1_error):
        """Add one sample of panoptic predictions and ground truths for
        evaluation.

        Args:
            semantics_pred (np.ndarray): Semantic predictions.
            semantics_gt (np.ndarray): Semantic ground truths.
            instances_pred (np.ndarray): Instance predictions.
            instances_gt (np.ndarray): Instance ground truths.
        """
        # get instance_class_id from instance_gt
        instance_class_ids = [self.num_classes - 1]
        for i in range(1, instances_gt.max() + 1):
            class_id = np.unique(semantics_gt[instances_gt == i])
            # assert class_id.shape[0] == 1, "each instance must belong to only one class"
            if class_id.shape[0] == 1:
                instance_class_ids.append(class_id[0])
            else:
                instance_class_ids.append(self.num_classes - 1)
        instance_class_ids = np.array(instance_class_ids)

        instance_count = 1
        final_instance_class_ids = []
        final_instances = np.zeros_like(instances_gt)  # empty space has instance id "0"

        for class_id in range(self.num_classes - 1):
            if np.sum(semantics_gt == class_id) == 0:
                continue

            if self.class_names[class_id] in ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'motorcycle', 'bicycle', 'pedestrian']:
                # treat as instances
                for instance_id in range(len(instance_class_ids)):
                    if instance_class_ids[instance_id] != class_id:
                        continue
                    final_instances[instances_gt == instance_id] = instance_count
                    instance_count += 1
                    final_instance_class_ids.append(class_id)
            else:
                # treat as semantics
                final_instances[semantics_gt == class_id] = instance_count
                instance_count += 1
                final_instance_class_ids.append(class_id)
                
        instances_gt = final_instances
        
        # avoid zero (ignored label)
        instances_pred = instances_pred + 1
        instances_gt = instances_gt + 1
        
        for j, threshold in enumerate(self.thresholds):
            tp_dist_mask = l1_error < threshold
            # for each class (except the ignored ones)
            for cl in self.include:
                # get a class mask
                pred_inst_in_cl_mask = semantics_pred == cl
                gt_inst_in_cl_mask = semantics_gt == cl

                # get instance points in class (makes outside stuff 0)
                pred_inst_in_cl = instances_pred * pred_inst_in_cl_mask.astype(int)
                gt_inst_in_cl = instances_gt * gt_inst_in_cl_mask.astype(int)

                # generate the areas for each unique instance prediction
                unique_pred, counts_pred = np.unique(
                    pred_inst_in_cl[pred_inst_in_cl > 0], return_counts=True)
                id2idx_pred = {id: idx for idx, id in enumerate(unique_pred)}
                matched_pred = np.array([False] * unique_pred.shape[0])

                # generate the areas for each unique instance gt_np
                unique_gt, counts_gt = np.unique(
                    gt_inst_in_cl[gt_inst_in_cl > 0], return_counts=True)
                id2idx_gt = {id: idx for idx, id in enumerate(unique_gt)}
                matched_gt = np.array([False] * unique_gt.shape[0])

                # generate intersection using offset
                valid_combos = np.logical_and(pred_inst_in_cl > 0,
                                            gt_inst_in_cl > 0)
                # add dist_mask
                valid_combos = np.logical_and(valid_combos, tp_dist_mask)

                id_offset_combo = pred_inst_in_cl[
                    valid_combos] + self.id_offset * gt_inst_in_cl[valid_combos]
                unique_combo, counts_combo = np.unique(
                    id_offset_combo, return_counts=True)

                # generate an intersection map
                # count the intersections with over 0.5 IoU as TP
                gt_labels = unique_combo // self.id_offset
                pred_labels = unique_combo % self.id_offset
                gt_areas = np.array([counts_gt[id2idx_gt[id]] for id in gt_labels])
                pred_areas = np.array(
                    [counts_pred[id2idx_pred[id]] for id in pred_labels])
                intersections = counts_combo
                unions = gt_areas + pred_areas - intersections
                ious = intersections.astype(float) / unions.astype(float)

                tp_indexes = ious > 0.5
                self.pan_tp[j][cl] += np.sum(tp_indexes)
                self.pan_iou[j][cl] += np.sum(ious[tp_indexes])

                matched_gt[[id2idx_gt[id] for id in gt_labels[tp_indexes]]] = True
                matched_pred[[id2idx_pred[id]
                            for id in pred_labels[tp_indexes]]] = True

                # count the FN
                if len(counts_gt) > 0:
                    self.pan_fn[j][cl] += np.sum(
                        np.logical_and(counts_gt >= self.min_num_points,
                                    ~matched_gt))

                # count the FP
                if len(matched_pred) > 0:
                    self.pan_fp[j][cl] += np.sum(
                        np.logical_and(counts_pred >= self.min_num_points,
                                    ~matched_pred))
    
    def count_pq(self):
        sq_all = self.pan_iou.astype(np.double) / np.maximum(
            self.pan_tp.astype(np.double), self.eps)
        rq_all = self.pan_tp.astype(np.double) / np.maximum(
            self.pan_tp.astype(np.double) + 0.5 * self.pan_fp.astype(np.double)
            + 0.5 * self.pan_fn.astype(np.double), self.eps)
        pq_all = sq_all * rq_all
        
        # mask classes not occurring in dataset
        mask = (self.pan_tp + self.pan_fp + self.pan_fn) > 0
        pq_all[~mask] = float('nan')

        table = PrettyTable([
            'Class Names',
            'RayPQ@%d' % self.thresholds[0],
            'RayPQ@%d' % self.thresholds[1],
            'RayPQ@%d' % self.thresholds[2]
        ])
        table.float_format = '.3'

        for i in range(len(self.class_names) - 1):
            table.add_row([
                self.class_names[i],
                pq_all[0][i], pq_all[1][i], pq_all[2][i],
            ], divider=(i == len(self.class_names) - 2))
        
        table.add_row([
            'MEAN',
            np.nanmean(pq_all[0]), np.nanmean(pq_all[1]), np.nanmean(pq_all[2])
        ])

        print(table)

        return {
            'RayPQ': np.nanmean(pq_all),
            'RayPQ@1': np.nanmean(pq_all[0]),
            'RayPQ@2': np.nanmean(pq_all[1]),
            'RayPQ@4': np.nanmean(pq_all[2]),
        }
