import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# from filterpy.kalman import KalmanFilter
from pic4people_tracking.utils.kalman_filter import KalmanFilter

np.random.seed(0)

def linear_assignment(cost_matrix):
  x, y = linear_sum_assignment(cost_matrix)
  return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
  """
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  """
  bb_gt = np.expand_dims(bb_gt, 0)
  bb_test = np.expand_dims(bb_test, 1)
  
  xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
  yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
  xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
  yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
    + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
  return(o)  

def l2_distance(detected_centroids, predicted_centroids):
    # Compute pairwise L2 distances
    # distances = np.sqrt(np.sum((detected_centroids[:, np.newaxis, :] - predicted_centroids[np.newaxis, :, :]) ** 2, axis=2))
    distances = cdist(detected_centroids, predicted_centroids, metric='euclidean')
    return distances


def combine_costs(iou_matrix, distance_matrix, weight_iou=0.3, weight_distance=0.7, distance_threshold=None):
    """
    Combine IoU and distance matrices into a single cost matrix for association.
    """
    print("iou matrix: ", iou_matrix)
    print("distance matrix: ", distance_matrix)
    
    iou_cost = 1 - iou_matrix
    
    # Normalize distances
    if distance_matrix.size > 1:
        min_dist = np.min(distance_matrix)
        max_dist = np.max(distance_matrix)
        if np.abs(max_dist - min_dist) > 1e-5:
          distance_normalized = (distance_matrix - np.min(distance_matrix)) / (np.max(distance_matrix) - np.min(distance_matrix))
          distance_cost = distance_normalized
        else:
          distance_cost = distance_matrix
    else:
        distance_cost = distance_matrix
        
    # Apply adaptive threshold to distance cost
    if distance_threshold is not None:
        distance_cost[distance_matrix > distance_threshold] = 1.0   
      
    print("iou cost: ", iou_cost)
    print("distance cost: ", distance_cost)
    
    # Combine costs
    total_cost = weight_iou * iou_cost + weight_distance * distance_cost
    print("total cost: ", total_cost)

    if distance_matrix.size == 0:
      new_distance_threshold = distance_threshold
    elif distance_matrix.size == 1:
      new_distance_threshold = max(0.5, distance_matrix[0, 0] * 2)
    else:
      new_distance_threshold = max(0.5, np.mean(distance_matrix) + 1.6*np.std(distance_matrix)) # + 2 * np.std(distance_matrix) 


    print("new distance threshold: ", new_distance_threshold)
    return total_cost, new_distance_threshold

def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w/2.
  y = bbox[1] + h/2.
  s = w * h    #scale is just area
  r = w / float(h)
  return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self, det, ind):
    """
    Initialises a tracker using initial bounding box + centroid.
    """
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=13, dim_z=7) 
    self.dt = 1.0/23 ## ~ 23 fps 
    self.kf.F = np.array([[1,0,0,0,0,0,0,self.dt,0,0,0,0,0],[0,1,0,0,0,0,0,0,self.dt,0,0,0,0],
                  [0,0,1,0,0,0,0,0,0,self.dt,0,0,0],[0,0,0,1,0,0,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0,self.dt,0,0],
                  [0,0,0,0,0,1,0,0,0,0,0,self.dt,0],[0,0,0,0,0,0,1,0,0,0,0,0,self.dt],[0,0,0,0,0,0,0,1,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,0,0,0,1,0,0],
                  [0,0,0,0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0,0,0,0],
                          [0,0,0,0,1,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0,0,0,0],[0,0,0,0,0,0,1,0,0,0,0,0,0]])

    self.kf.R[2:4,2:4] *= 10.
    self.kf.R[4:7,4:7] *= 10. ##### measured position
    self.kf.P[7:,7:] *= 1000. # give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    self.kf.Q[9,9] *= 0.01
    self.kf.Q[7:10,7:10] *= 0.01 #####
    self.kf.Q[4:7,4:7] = 0.3
    self.kf.Q[10:,10:] *= 0.6 ##### the lower, the smoother the trajectories

    z_box = convert_bbox_to_z(det[:4])
    z_centroid = np.array(det[4:7]).reshape(3,1)
    self.kf.x[:7] = np.append(z_box,z_centroid).reshape(7,1)
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0
    self.det_indx = ind

  def update(self,det):
    """
    Updates the state vector with observed det.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    z_box = convert_bbox_to_z(det[:4])
    z_centroid = np.array(det[4:7]).reshape(3,1)
    self.kf.update(np.append(z_box,z_centroid).reshape(7,1))

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box + centroid estimate.
    """
    if((self.kf.x[9]+self.kf.x[2])<=0):
      self.kf.x[9] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    bbox = convert_x_to_bbox(self.kf.x)
    self.history.append(np.append(bbox,self.kf.x[4:7].reshape(1,-1)))
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box + centroid estimate.
    """
    bbox = convert_x_to_bbox(self.kf.x)
    return np.append(bbox,self.kf.x[4:7].reshape(1,-1))
  
  def get_3Dvelocity(self):
    return self.kf.x[10:13].reshape(1,-1)


def associate_detections_to_trackers(detections,trackers,cost_threshold = 0.6, distance_threshold=None):
  """
  Assigns detections to tracked object (both represented as bounding boxe + centroid)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  print("num. trackers: ", len(trackers))
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,8),dtype=int), None

  iou_matrix = iou_batch(detections[:,0:4], trackers[:,0:4])
  distance_matrix = l2_distance(detections[:,4:7], trackers[:,4:7])
  cost_matrix, new_distance_threshold = combine_costs(iou_matrix, distance_matrix, distance_threshold=distance_threshold)

  #######
  if min(cost_matrix.shape) > 0:
    a = (cost_matrix < cost_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(cost_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))

  unmatched_detections = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(cost_matrix[m[0], m[1]]>cost_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  print("matches: ", matches)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers), new_distance_threshold
  ######

class SortRGBD(object):
  def __init__(self, max_age=1, min_hits=3, cost_threshold=0.6):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.cost_threshold = cost_threshold
    self.distance_threshold = None
    self.trackers = []
    self.frame_count = 0

  def update(self, dets=np.empty((0, 8))):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,x,y,z,score],[x1,y1,x2,y2,x,y,z,score],...]

    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 8)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    # get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers), 8))
    to_del = []
    ret = []
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict()
      trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6], 0]
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    matched, unmatched_dets, unmatched_trks, new_distance_threshold = associate_detections_to_trackers(dets,
                                                                                trks, self.cost_threshold, self.distance_threshold)
    self.distance_threshold = new_distance_threshold

    # update matched trackers with assigned detections
    for m in matched:
      self.trackers[m[1]].update(dets[m[0], :])
      self.trackers[m[1]].det_indx = m[0]

    # create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,:],i)
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()
        if (trk.time_since_update < 4) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits): #(trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
          ind = trk.det_indx
          ret.append(np.concatenate((d,[trk.id+1],[ind])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        # remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,9))
  
  def get_3Dvelocities(self):
    """
    return the velocity of the currently active trackers. Assumed to be called after update method.
    """
    ret = []

    for trk in reversed(self.trackers):
        d = trk.get_3Dvelocity()[0]
        if (trk.time_since_update < 4) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits):
          ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,4))
  
  def get_reliabilities(self):
    """
    return a simple reliability score based on the residual of the currently active trackers. 
    Assumed to be called after update method.
    """
    ret = []

    for trk in reversed(self.trackers):
        reliability = max(0.0, min(1.0, 1.0 - trk.kf.normalized_residual))
        if (trk.time_since_update < 4) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits):
          ret.append(np.array([reliability, trk.id+1]).reshape(1,-1))
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,2))
  
