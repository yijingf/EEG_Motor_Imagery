import numpy as np
from hypertools.tools.procrustes import procrustes

class hyperalignment():
    
    def __init__(self, ):
        
        return
    
    def train(self, data):
        """
        data: list of array, subject, run * feat
        """
        sizes_0 = [x.shape[0] for x in data]
        sizes_1 = [x.shape[1] for x in data]

        #find the smallest number of rows
        self.R = min(sizes_0)
        self.C = max(sizes_1)
        
        m = [np.empty((self.R, self.C), dtype=np.ndarray)] * len(data)
        
        for idx,x in enumerate(data):
            y = x[0:self.R,:]
            missing = self.C - y.shape[1]
            add = np.zeros((y.shape[0], missing))
            y = np.append(y, add, axis=1)
            m[idx]=y
            
        ##STEP 1: TEMPLATE##
        for x in range(0, len(m)):
            if x==0:
                template = np.copy(m[x])
            else:
                next = procrustes(m[x], template / (x + 1))
                template += next
        template /= len(m)
        
        ##STEP 2: NEW COMMON TEMPLATE##
        #align each subj to the template from STEP 1
        template2 = np.zeros(template.shape)
        for x in range(0, len(m)):
            next = procrustes(m[x], template)
            template2 += next
        template2 /= len(m)
        self.template = template2
                
        #STEP 3 (below): ALIGN TO NEW TEMPLATE
        aligned = [np.zeros(template2.shape)] * len(m)
        for x in range(0, len(m)):
            next = procrustes(m[x], template2)
            aligned[x] = next
        return aligned
    
    def transform(self, data):
        for idx,x in enumerate(data):
            y = x[0:self.R,:]
            missing = self.C - y.shape[1]
            add = np.zeros((y.shape[0], missing))
            y = np.append(y, add, axis=1)
            m[idx]=y
        
        aligned = [np.zeros(self.template2.shape)] * len(m)
        for x in range(0, len(m)):
            next = procrustes(m[x], self.template2)
            aligned[x] = next
        return aligned