# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

from pic4people_tracking.StrongSORT.motion.cmc.ecc import ECC
from pic4people_tracking.StrongSORT.motion.cmc.orb import ORB
from pic4people_tracking.StrongSORT.motion.cmc.sift import SIFT
from pic4people_tracking.StrongSORT.motion.cmc.sof import SOF


def get_cmc_method(cmc_method):
    if cmc_method == 'ecc':
        return ECC
    elif cmc_method == 'orb':
        return ORB
    elif cmc_method == 'sof':
        return SOF
    elif cmc_method == 'sift':
        return SIFT
    else:
        return None
