import json

class Omni3DDataset():

    def __init__(self, info, images, categories, annotations):
        self.info = info
        self.images = images
        self.categories = categories
        self.annotations = annotations


    def toJSON(self):
         return json.dumps(self, default=vars, sort_keys=True, indent=4)


class Omni3DObject():

    uid = 0


    def __init__(self, dataset_id, image_id, seq_id, category_id, category_name, \
                            valid3D, bbox2D_tight, bbox2D_proj, bbox2D_trunc, bbox3D_cam, center_cam, dimensions, \
                            R_cam, score, behind_camera=-1, visibility=-1, truncation=-1, segmentation_pts=-1, lidar_pts=-1, depth_error=-1):

        self.id = Omni3DObject.uid
        Omni3DObject.uid += 1
        self.dataset_id = dataset_id
        self.image_id = image_id
        self.seq_id = seq_id
        self.category_id = category_id
        self.category_name = category_name
        self.valid3D = valid3D
        self.bbox2D_tight = bbox2D_tight
        self.bbox2D_proj = bbox2D_proj
        self.bbox2D_trunc = bbox2D_trunc
        self.bbox3D_cam = bbox3D_cam
        self.center_cam = center_cam
        self.dimensions = dimensions
        self.R_cam = R_cam
        self.score = score
        self.behind_camera = behind_camera
        self.visibility = visibility
        self.truncation = truncation
        self.segmentation_pts = segmentation_pts
        self.lidar_pts = lidar_pts
        self.depth_error = depth_error

    def toJSON(self):
        return json.dumps(self, default=vars, sort_keys=True, indent=4)


class Omni3DImage():

    def __init__(self, id_, dataset_id, seq_id, width, height, file_path, K, gt, t, src_90_rotate=0, src_flagged=0):

        self.id = id_
        self.dataset_id = dataset_id
        self.seq_id = seq_id
        self.width = width
        self.height = height
        self.file_path = file_path
        self.K = K
        self.src_90_rotate = src_90_rotate
        self.src_flagged = src_flagged
        self.gt = gt
        self.t = t

    def toJSON(self):
        return json.dumps(self, default=vars, indent=4)


class Omni3DCategory():

    def __init__(self, id_, name, supercategory=""):

        self.id = id_
        self.name = name
        self.supercategory = supercategory

    def toJSON(self):
        return json.dumps(self, default=vars, sort_keys=True, indent=4)


class Omni3DInfo():
    def __init__(self, id_, source, name, split, version, url):
        self.id = id_
        self.source = source
        self.name = name
        self.split = split
        self.version = version
        self.url = url


    def toJSON(self):
        return json.dumps(self, default=vars, sort_keys=True, indent=4)

