from alignment_calibration import runit
from spotchecker import datamaker
import os
import glob
import pandas as pd

option = 3
# path = './RawDataFromScottie/Monochrome/g/'
path = './data/'
extension = 'czi'
os.chdir(path)
result = glob.glob('*.{}'.format(extension))
print(result)

listofframes = []
listofredframes = []
listofgreenframes = []
listoffarredframes = []

counter = 1
for item in result:
    print(item)
    try:
        runit(item)
        try:
            if option == 3:
                data, redframe, greenframe, farredframe = datamaker(option)
            if option == 2:
                data, redframe, greenframe = datamaker(option)
            data = data.rename({'Percentage': item[:-4]}, axis=1)
            print(data)
            if counter == 1:
                listofframes.append(data)
                listofredframes.append(redframe)
                listofgreenframes.append(greenframe)
                if option == 3:
                    listoffarredframes.append(farredframe)
            else:
                listofframes.append(data[item[:-4]])
            print("check3")
            counter = counter + 1
        except:
            print("analysis failed")
            pass
    except:
        print("Failed alignment")
        pass

print(listofframes)
final_data = pd.concat(listofframes, axis=1)
final_data_red = pd.concat(listofredframes, axis=1)
final_data_green = pd.concat(listofgreenframes, axis=1)
if option == 3:
    final_data_farred = pd.concat(listoffarredframes, axis=1)
print(final_data)

final_data.to_csv("finaldata2.csv")
final_data_red.to_csv("finaldata2g.csv")
final_data_green.to_csv("finaldata2r.csv")
if option == 3:
    final_data_farred.to_csv("finaldata2fr.csv")
