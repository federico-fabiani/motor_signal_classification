class ObjectSelector:
    def __init__(self):
        self.objects_dict = {
            "mixed": ["11", "12", "13", "14", "15", "16"],
            "rings": ["21", "22", "23", "24", "25", "26"],
            "cubes": ["31", "32", "33", "34", "35", "36"],
            "balls": ["41", "42", "43", "44", "45", "46"],
            "hCylinder": ["51", "52", "53", "54", "55", "56"],
            "boxes": ["61", "62", "63", "64", "65", "66"],
            "vCylinder": ["71", "72", "73", "74", "75", "76"],
            "special": ["81", "82", "83", "84", "85", "86"],
            "precision": "0",
            "strength": "1"
        }

    def get_dict(self):
        return self.objects_dict

    def get_shape(self, shape):
        if shape in self.objects_dict.keys():
            return self.objects_dict[shape]
        else:
            return None

    def get_non_special(self):
        result = []
        labels = []
        for k in self.objects_dict.keys():
            if k not in ['mixed', 'special', 'strength', 'precision']:
                result.append(self.objects_dict[k])
                labels.append(k)
        return result, labels

    # def get_size(self, size):
    #     for k in self.objects_dict.keys():
    #         result.append()