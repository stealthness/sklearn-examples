from sw_data_loader.SWData import SWData
import sw_path.WORK_ROOT as ROOT
print("run test")
data = SWData()
data.load_img_datafiles(ROOT + "RES/TestFolder")
print(data.base_path)
print((len(data.data_classes)))
print(data.is_datafiles_all_same_dim())