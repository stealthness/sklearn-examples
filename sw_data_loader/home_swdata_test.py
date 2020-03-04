from sw_data_loader.SWDataClassFile import SWData

print("run home")
data = SWData()
data.load_img_datafiles("I:/RES/TestFolder")
print(data.base_path)
print((len(data.data_classes)))

print(data.is_datafiles_all_same_dim())