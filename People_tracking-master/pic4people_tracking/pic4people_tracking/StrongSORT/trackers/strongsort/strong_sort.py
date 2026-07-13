
import numpy as np
import time

# from pic4people_tracking.StrongSORT.appearance.reid_auto_backend import ReidAutoBackend
from pic4people_tracking.StrongSORT.trackers.strongsort.reid_openvino import ReIDInference
from pic4people_tracking.StrongSORT.motion.cmc.ecc import ECC
from pic4people_tracking.StrongSORT.trackers.strongsort.sort.detection import Detection
from pic4people_tracking.StrongSORT.trackers.strongsort.sort.tracker import Tracker
from pic4people_tracking.StrongSORT.utils.matching import NearestNeighborDistanceMetric
from pic4people_tracking.StrongSORT.utils.ops import xyxy2tlwh, tlwh2xyxy


class StrongSORT(object):
    def __init__(
        self,
        model_weights,
        device,
        fp16,
        per_class=False,
        max_dist=0.3,
        max_l2_dist=0.5,
        max_age=30,
        n_init=10,
        nn_budget=100,
        mc_lambda=0.9, #0.995,
        ema_alpha=0.9,
    ):

        self.per_class = per_class
        # rab = ReidAutoBackend(
        #     weights=model_weights, device=device, half=fp16
        # )
        # self.model = rab.get_backend()
        self.model = ReIDInference(weights=model_weights)
        self.tracker = Tracker(
            metric=NearestNeighborDistanceMetric("cosine", max_dist, nn_budget),
            max_l2_dist=max_l2_dist,
            max_age=max_age,
            n_init=n_init,
            mc_lambda=mc_lambda,
            ema_alpha=ema_alpha,
        )
        # self.cmc = ECC()

    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None, dt: float = 1./18) -> np.ndarray:
        assert isinstance(
            dets, np.ndarray
        ), f"Unsupported 'dets' input format '{type(dets)}', valid format is np.ndarray"
        assert isinstance(
            img, np.ndarray
        ), f"Unsupported 'img' input format '{type(img)}', valid format is np.ndarray"
        assert (
            len(dets.shape) == 2
        ), "Unsupported 'dets' dimensions, valid number of dimensions is two"
        assert (
            dets.shape[1] == 8
        ), "Unsupported 'dets' 2nd dimension lenght, valid lenghts is 6"

        dets = np.hstack([dets, np.zeros((dets.shape[0],1)), 
                          np.arange(len(dets)).reshape(-1, 1)])
        xyxy = dets[:, 0:4]
        xyz = dets[:,4:7]
        confs = dets[:, 7]
        clss = dets[:,8].astype("int") # 0: Person
        det_ind = dets[:, 9].astype("int")

        # if len(self.tracker.tracks) >= 1:
        #     warp_matrix = self.cmc.apply(img, xyxy)
        #     for track in self.tracker.tracks:
        #         track.camera_update(warp_matrix)

        # extract appearance information for each detection
        if embs is not None:
            features = embs
        else:
            start_t = time.perf_counter()
            features = self.model.get_features(xyxy, img)
            print("Re-ID time: ", time.perf_counter()-start_t)

        tlwh = xyxy2tlwh(xyxy)
        detections = [
            Detection(box, pos3d, conf, cls, det_ind, feat) for
            box, pos3d, conf, cls, det_ind, feat in
            zip(tlwh, xyz, confs, clss, det_ind, features)
        ]

        # update tracker
        self.tracker.predict(dt)
        start_u = time.perf_counter()
        self.tracker.update(detections)
        print("update time: ", time.perf_counter()-start_u)


        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            print("track id: ",track.id)
            if not (track.is_confirmed() and track.time_since_update < 1):
                continue
            
            bbox = track.bbox
            x1, y1, x2, y2 = tlwh2xyxy(bbox) # track.to_tlbr()
            x, y, z = track.mean[0:3]

            id = track.id
            # conf = track.conf
            # cls = track.cls
            det_ind = track.det_ind

            outputs.append(
                np.concatenate(([x1, y1, x2, y2], [x, y, z], [id], [det_ind])).reshape(1, -1) # [conf], [cls], 
            )

        if len(outputs) > 0:
            tracks = np.concatenate(outputs)
            return tracks
        return np.empty((0,9))
        return np.array([])
    
    def get_3Dvelocities(self):
        """
        return the velocity of the currently active trackers. Assumed to be called after update method.
        """
        ret = []
    
        for track in self.tracker.tracks:
            if not (track.is_confirmed() and track.time_since_update < 1):
                continue
  
            vx, vy, vz = track.mean[3:6]
    
            id = track.id
            conf = track.conf
            cls = track.cls
            det_ind = track.det_ind
    
            ret.append(
                np.concatenate(([vx, vy, vz], [id], [det_ind])).reshape(1, -1) # [conf], [cls], 
            )
      
        if len(ret) > 0:
            tracks = np.concatenate(ret)
            return tracks
        return np.empty((0,5))
    
    def get_reliabilities(self):
        """
        return the velocity of the currently active trackers. Assumed to be called after update method.
        """
        ret = []
    
        for track in self.tracker.tracks:
            if not (track.is_confirmed() and track.time_since_update < 1):
                continue
            rel = max(0.0, min(1.0, 1.0 - track.kf.normalized_residual))
            id = track.id
            det_ind = track.det_ind
            ret.append(
                np.concatenate(([rel], [id], [det_ind])).reshape(1, -1) # [conf], [cls], 
            )
        if len(ret) > 0:
            tracks = np.concatenate(ret)
            return tracks
        return np.empty((0,3))
