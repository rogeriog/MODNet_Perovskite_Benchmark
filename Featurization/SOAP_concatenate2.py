import pickle
import numpy as np
SOAP_upto30=pickle.load(open("SOAP_matrix_upto30.pkl","rb"))
SOAPfinal=pickle.load(open("SOAP_matrix_final.pkl","rb"))
SOAP_total=np.vstack((SOAP_upto30,SOAPfinal))
SOAP_total=np.unique(SOAP_total,axis=0)
SOAP_total=SOAP_total[1:]
print(SOAP_total.shape)
pickle.dump(SOAP_total,open("SOAP_total.pkl","wb"))
