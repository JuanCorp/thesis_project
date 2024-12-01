import json


class DataSaver(object):

    def __init__(self) -> None:
        pass


    def save_object(self,obj,path):
        with open(path,"w") as save_path:
            json.dump(obj,save_path)