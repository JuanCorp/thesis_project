import json
import numpy as np

class NumpyFloatValuesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)



class DataSaver(object):

    def __init__(self) -> None:
        pass


    def save_object(self,obj,path):
        with open(path,"w") as save_path:
            json.dump(obj,save_path,cls=NumpyFloatValuesEncoder)