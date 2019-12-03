import numpy as np

class procrustes():
    """
    Function to project from one space to another using Procrustean
    transformation (shift + scaling + rotation + reflection).
    Parameters
    ----------
    source : Numpy array
        Array to be aligned to target's coordinate system.
    target: Numpy array
        Source is aligned to this target space
    Returns
    ----------
    aligned_source : Numpy array
        The array source is aligned to target and returned
    """
    def __init__(self, proj=None):
        self.proj = proj

    def __call__(self, source, target):
        self.proj = self.fit(source, target)
        return self.transform(source)

    def fit(self, source, target):

        datas = (source, target)
        sn, sm = source.shape
        tn, tm = target.shape

        # Sums of squares
        ssqs = [np.sum(d**2, axis=0) for d in datas]

        norms = [ np.sqrt(np.sum(ssq)) for ssq in ssqs ]
        normed = [ data/norm for (data, norm) in zip(datas, norms) ]

        source, target = normed

        # Orthogonal transformation
        # figure out optimal rotation
        U, s, Vh = np.linalg.svd(np.dot(target.T, source),
                                 full_matrices=False)
        T = np.dot(Vh.T, U.T)

        ss = sum(s)

        # Assign projection
        scale = ss * norms[1] / norms[0]
        proj = scale * T

        return proj

    def transform(self, data):
        if self.proj is None:
            raise ValueError("Run .fit before transform")
        d = np.asmatrix(data)

        # Do projection
        res = (d * self.proj).A

        return res

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
        p1 = procrustes()
        for x in range(0, len(m)):
            if x==0:
                template = np.copy(m[x])
            else:
                next = p1(m[x], template / (x + 1))
                template += next
        template /= len(m)
        
        ##STEP 2: NEW COMMON TEMPLATE##
        #align each subj to the template from STEP 1
        p2 = procrustes()
        template2 = np.zeros(template.shape)
        for x in range(0, len(m)):
            next = p2(m[x], template)
            template2 += next
        template2 /= len(m)
        self.template = template2
                
        #STEP 3 (below): ALIGN TO NEW TEMPLATE
        mappers = [procrustes() for _ in m]
#         aligned = [np.zeros(template2.shape)] * len(m)
        for x in range(0, len(m)):
            next = mappers[x](m[x], template2)
#             aligned[x] = next
        return mappers
#         return aligned, mappers