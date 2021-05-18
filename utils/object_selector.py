"""ObjectSelector class embed a dictionary associating single object to a common shape, and it is useful to make it
flexible to select which objects or shapes to use as classes for the model. Each method return a list where every entry
is the element or list of elements, composing a class. Besides, a list of names for those classes is provided"""


class ObjectSelector:
    def __init__(self):
        self.objects_dict = {
            "mixed": ["11", "12", "13", "14", "15", "16"],
            "rings": ["21", "22", "23", "24", "25", "26"],
            "cubes": ["31", "32", "33", "34", "35", "36"],
            "balls": ["41", "42", "43", "44", "45", "46"],
            "hCylinder": ["51", "52", "53", "54", "55", "56", "57", "58"],  # 57 and 58 are 2 different ways to grasp 56
            "boxes": ["61", "62", "63", "64", "65", "66"],
            "vCylinder": ["71", "72", "73", "74", "75", "76"],
            "special": ["81", "82", "83", "84", "85", "86"],
            "special2": ["91", "92", "93", "94", "95", "96"],  # not clear labels found on one dataset
            "precision": "0",
            "strength": "1"
        }

    def get_dict(self):
        return self.objects_dict

    def get_shapes(self, shapes, group_labels=False):
        result = []
        names = []
        if type(shapes) != list:
            print(f'Input element {shapes} casted to list')
            shapes = [shapes]
        for shape in shapes:
            if shape in self.objects_dict.keys():
                if group_labels:
                    result.append(self.objects_dict[shape])
                    names.append(shape)
                else:
                    for obj in self.objects_dict[shape]:
                        result.append(obj)
                        names.append(obj)
            else:
                print(f'{shape} not present in objects dict; try with {self.objects_dict.keys()}')
        return list(zip(result, names))

    def get_non_special(self, group_labels=False, include_mixed=False):
        result = []
        names = []
        excluded = ['mixed', 'special', 'special2', 'strength', 'precision']
        if include_mixed:
            excluded.pop(excluded.index('mixed'))
        for k in self.objects_dict.keys():
            if k not in excluded:
                if group_labels:
                    result.append(self.objects_dict[k])
                    names.append(k)
                else:
                    for obj in self.objects_dict[k]:
                        result.append(obj)
                        names.append(obj)
        return list(zip(result, names))

    def get_all(self, group_labels=False):
        result = []
        names = []
        for k in self.objects_dict.keys():
            if group_labels:
                result.append(self.objects_dict[k])
                names.append(k)
            else:
                for obj in self.objects_dict[k]:
                    result.append(obj)
                    names.append(obj)
        return list(zip(result, names))


if __name__ == '__main__':
    os = ObjectSelector()
    # classes, clNames = os.get_all(group_labels=False)
    classes, clNames = os.get_shapes('rings', group_labels=True)
    print(f'Number of classes: {len(classes)}')
    for pair in zip(classes, clNames):
        print(pair)