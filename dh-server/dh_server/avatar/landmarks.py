"""Landmark indices shared by the MuseTalk preparation path.

MuseTalk's own `get_landmark_and_bbox` derives its crop from DWPose's
COCO-WholeBody 68-point face subset (iBUG-68 order), using:

    face_land_mark[29]              -> nose
    max(face_land_mark[:, 1])       -> chin (lowest of the 68)
    min/max of face_land_mark[:, 0] -> face width

We get the same geometry from mediapipe FaceMesh's 468-point mesh, so these are
the mesh indices carrying the equivalent semantics. Kept in one place because
`spike/detect.py` and `server/avatar/musetalk_pipeline.py` must agree -- if they
drift, the spike stops proving anything about production.
"""

NOSE_TIP = 4
CHIN = 152
LEFT_CHEEK = 234
RIGHT_CHEEK = 454
