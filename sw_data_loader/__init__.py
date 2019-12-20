import os
from PIL import Image


class SWData:
    data_img = {}
    data_class = {}
    base_path = ""
    data_classes = set()
    data_images_equal_size = False

    def __int__(self):
        pass

    def load_img_datafiles(self, folder_location):
        """
        Creates a dictionary of filenames, classification and location
        """
        self.base_path = folder_location
        self.data_classes = os.listdir(folder_location)
        for data_class in self.data_classes:
            for file in os.listdir(self.base_path + "/" + data_class):
                self.data_class[self.base_path + "/" + data_class + "/" + file] = data_class

    def load_image_data(self):
        """
        Load the image data
        """
        for data_class in self.data_classes:
            for file in os.listdir(self.base_path + "/" + data_class):
                self.data[self.base_path + "/" + data_class + "/" + file] = Image.open()

    def is_datafiles_all_same_dim(self):
        img_size = (-1, -1)
        for file in self.data_class:
            img = Image.open(file)
            if img_size == (-1, -1):
                img_size = img.size
                img.close()
            elif img_size != img.size:
                return False
            else:
                img.close()
        return True

    def clear(self):
        self.data_class = {}
        self.base_path = ""
        self.data_classes = []


print("run test")
data = SWData()
data.load_img_datafiles("I:/RES/TestFolder")
print(data.base_path)
print((len(data.data_classes)))

print(data.is_datafiles_all_same_dim())
