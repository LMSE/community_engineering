#https://cobrapy.readthedocs.io/en/latest/io.html


from __future__ import print_function

import os
import cobra

os.chdir("models")

matlab_model = cobra.io.load_matlab_model("iAF1260.mat")
cobra.io.save_json_model(matlab_model, "iAF1260.json")
matlab_model = cobra.io.load_matlab_model("iYO844.mat")
cobra.io.save_json_model(matlab_model, "iYO844.json")
matlab_model = cobra.io.load_matlab_model("iML1515.mat")
cobra.io.save_json_model(matlab_model, "iML1515.json")




