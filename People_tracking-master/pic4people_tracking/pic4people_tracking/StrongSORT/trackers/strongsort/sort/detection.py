
class Detection(object):
    """
    This class represents a bounding box detection plus 3D coordinates
    in a single image.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    xyz : array_like
        3D coordinates '(x, y, z)'
    conf : float
        Detector confidence score.
    cls : int
        Class of the detected object.
    det_ind : int
        Index of the detected object.    
    feat : array_like
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    xyz : array_like
        3D coordinates '(x, y, z)'
    conf : float
        Detector confidence score.
    cls : int
        Class of the detected object.
    det_ind : int
        Index of the detected object.    
    feat : array_like
        A feature vector that describes the object contained in this image.

    """

    def __init__(self, tlwh, xyz, conf, cls, det_ind, feat):
        self.tlwh = tlwh
        self.xyz = xyz
        self.conf = conf
        self.cls = cls
        self.det_ind = det_ind
        self.feat = feat

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
