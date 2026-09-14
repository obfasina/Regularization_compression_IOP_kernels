

import numpy as np
import math

class haar_paraproduct:

    def __init__(self,data,orig):

        """
        orig = smooth composition of Holder data
        data = holder data
        """

        # Define Holder data
        self.f = data
        self.Af = orig
        
        return


    def haar(self,j,npts):
    
        # create haar function at scale j
        dx = int(npts/(2**j))
        dxd2 = int(dx/2)
        left = np.ones(dxd2).reshape(1,dxd2)
        left = left.reshape(1,dxd2)
        right = np.ones(dxd2)*-1
        right = right.reshape(1,dxd2)
        haar = np.squeeze(np.concatenate((left,right),axis=1))
        nhaar = haar/np.linalg.norm(haar)
        
        # Generate basis for subspace at scale j using location parameter
        veclist = []
        for k in range(0,npts,dx):
            vec = np.zeros((npts))
            vec[k:k + dx] = nhaar
            veclist.append(vec)
        veclist = np.array(veclist)
    
        return veclist


    def scl(self,j,npts):
    
        # Create scaling function at scale j
        dx = int(npts/(2**j))
        scl = np.ones(dx)
        nscl = scl/np.linalg.norm(scl)
        
        # Generate basis for subspace at scale j using location parameter
        veclist = []
        for k in range(0,npts,dx):
            vec = np.zeros((npts))
            vec[k:k + dx] = nscl
            veclist.append(vec)
        veclist = np.array(veclist)
    
        return veclist
    

    def genconvs(self,jx,jy):

        NY = self.f.shape[0]
        NX = self.f.shape[1]
        
        Wx = self.haar(jx,NX)
        Wy = self.haar(jy,NY)
        Vx = self.scl(jx,NX)
        Vy = self.scl(jy,NY)
        
        nyloc = Vy.shape[0]
        nxloc = Vx.shape[0]
        
        convopWxWy = np.ones((NY,NX))
        convopWxVy = np.ones((NY,NX))
        convopVxWy = np.ones((NY,NX))
        convopVxVy = np.ones((NY,NX)) 
        
        
        for kx in range(0,nxloc):
            for ky in range(0,nyloc):
                
                # NOTE: Because of disjoint support, row and column indices completely fill coefficient matrices at a fixed scale
                
                # NOTE: I don't think coefficient matrices should be normalized again, since the haar and scaling functions are orthonormal
                
                tens = np.kron(Wx[kx,:].reshape(1,NX),Wy[ky,:].reshape(NY,1))
                rowidx, colidx = np.nonzero(tens)
                suppsize = len(rowidx)*len(colidx)
                coef = np.sum(self.f[rowidx,colidx] * tens[rowidx,colidx])
                convopWxWy[rowidx,colidx] = coef
        
                tens = np.kron(Wx[kx,:].reshape(1,NX),Vy[ky,:].reshape(NY,1))
                rowidx, colidx = np.nonzero(tens)
                suppsize = len(rowidx)*len(colidx)
                coef = np.sum(self.f[rowidx,colidx] * tens[rowidx,colidx])
                convopWxVy[rowidx,colidx] = coef
        
                tens = np.kron(Vx[kx,:].reshape(1,NX),Wy[ky,:].reshape(NY,1))
                rowidx, colidx = np.nonzero(tens)
                suppsize = len(rowidx)*len(colidx)
                coef = np.sum(self.f[rowidx,colidx] * tens[rowidx,colidx])
                convopVxWy[rowidx,colidx] = coef
        
                tens = np.kron(Vx[kx,:].reshape(1,NX),Vy[ky,:].reshape(NY,1))
                rowidx, colidx = np.nonzero(tens)
                suppsize = len(rowidx)*len(colidx)
                coef = np.sum(self.f[rowidx,colidx] * tens[rowidx,colidx])
                convopVxVy[rowidx,colidx] = coef          
                
    
        return convopWxWy, convopWxVy, convopVxWy, convopVxVy


    def para_approx(self,mNjx,mNjy):


        # Specify number of points
        NY = self.f.shape[0]
        NX = self.f.shape[1]
        App = np.zeros((NY,NX))
        if self.kernel == 'schrod':
            App = np.zeros((NY,NX),dtype=complex)

        # Starting number of scales
        smNjx = 3
        smNjy = 3

        for jdx in range(smNjx,mNjx):
            for jdy in range(smNjy,mNjy):
        
                WxWy, WxVy, VxWy, VxVy = self.genconvs(jdx,jdy)
                VxVy = np.where(VxVy == 0, 1e-10, VxVy) 


                # Nonlinearity (Heat Kernel)
                if self.kernel == 'heat':
                    escl = 0.2
                    Ap = -escl*np.exp(-escl*VxVy)
                    Adp = escl*escl*np.exp(-escl*VxVy)

                # Nonlinearity (Potential)
                if self.kernel == 'pot':
                    Ap = VxVy**(-1)
                    Adp = VxVy**(-2)

                # Nonlinearity (Parametrix)
                if self.kernel == 'parametrix':
                    Ap = -0.5*VxVy**(-1.5)
                    Adp = 0.75*VxVy**(-2.5)

                if self.kernel == 'log':

                    # Constant Coefficient
                    #Ap = VxVy**(-1)
                    #Adp = VxVy**(-2)

                    # NOTE: if the degree of the polynomial is too high you could get infinite values (well behaved for up to degree 25)

                    # Variable coefficient 
                    termone = -(self.deg + 1)*self.coeffn*self.eps**(-1)*(VxVy**(-(self.deg+2)))
                    termtwo = (self.coeffn*(VxVy**(-(self.deg+1)))*self.eps**(-1)) - (self.coeffzro*self.eps**(-1))
                    Ap = termone/ (termtwo + 1e-10)
                    Adp = ((self.eps - 1)*(self.deg**2 + 2*self.deg + 2)*VxVy**(-2)) + (self.deg*VxVy**(-2)) - (self.coeffzro*(self.coeffn**(-1))*(VxVy**(self.deg + 1)))


                if self.kernel == 'schrod':
                    #Ap = np.exp((1j*(VxVy**2))/self.t)*(((2*1j)/self.t)*(VxVy))
                    #Adp = np.exp((1j*(VxVy**2))/self.t)*(-4*(VxVy**2)/(self.t**2))

                    Ap = 1j*np.exp(1j*VxVy) 
                    Adp = -np.exp(1j*VxVy) 

                # Build approximation
                frst = Ap * WxWy
                scnd = Adp * WxVy * VxWy
                App = App + frst + scnd

        self.App = App


        return App
        

    def twave_approx(self,data,mNjx,mNjy):

        NY = data.shape[0]
        NX = data.shape[1]
        Afrecon = np.zeros((NY,NX))
        self.coefs = []
        self.norms = []
        self.supp = []
        self.wvtens = []
        self.coefs_hld = []
        self.supp_hld = []


        # Storage list
        self.wv = []
        self.wvsc = []
        self.scwv = []
        self.sc = []

        self.wvcoef = []
        self.wvscoef = []
        self.scwvcoef = []
        self.scoef = []
        
        self.wvcoefnm = []
        self.wvscoefnm = []
        self.scwvcoefnm = []
        self.scoefnm = []


        
        for jxi in range(0,mNjx):
            for jyi in range(0,mNjy):
        
                hxv = self.haar(jxi,NX) 
                hyv = self.haar(jyi,NY)
                sxv = self.scl(jxi,NX)
                syv = self.scl(jyi,NY)

                
                nxloc = hxv.shape[0]
                nyloc = hyv.shape[0]
                dnm = (2**(-(jxi + jyi)))**(self.alf + 0.5)
                supp = (2**(-(jxi + jyi)))

                
                suppcoefs = [] # Compute coefficients for each support size
                for kx in range(nxloc):
                    for ky in range(nyloc):                    

                        # Compute tensor wavelets
                        wvtens = np.kron(hxv[kx,:].reshape(1,NX),hyv[ky,:].reshape(NY,1))
                        wvsctens = np.kron(hxv[kx,:].reshape(1,NX),syv[ky,:].reshape(NY,1))
                        scwvtens = np.kron(sxv[kx,:].reshape(1,NX),hyv[ky,:].reshape(NY,1))
                        sctens = np.kron(sxv[kx,:].reshape(1,NX),syv[ky,:].reshape(NY,1))
                        self.wv.append(wvtens)
                        self.wvsc.append(wvsctens)
                        self.scwv.append(scwvtens)
                        self.sc.append(sctens)

                        
                        # Compute coefficients
                        wvcoef = np.sum(data * wvtens)
                        wvscoef = np.sum(data * wvsctens)
                        scwvcoef = np.sum(data * scwvtens)
                        scoef = np.sum(data * sctens)
                        
                        self.wvcoef.append(wvcoef)
                        self.wvscoef.append(wvscoef)
                        self.scwvcoef.append(scwvcoef)
                        self.scoef.append(scoef)
                        
                        self.wvcoefnm.append(wvcoef/supp)
                        self.wvscoefnm.append(wvscoef/supp)
                        self.scwvcoefnm.append(scwvcoef/supp)
                        self.scoefnm.append(scoef/supp)

                        
                        # Old computations (saved for repeatability)
                        acoef = np.abs(np.sum(data * wvtens))
                        self.norms.append(acoef/dnm)
                        Afrecon += np.sum(data * wvtens) * wvtens
                        self.coefs.append(acoef)
                        self.supp.append(supp)
                        suppcoefs.append(acoef)


                        
                self.coefs_hld.append(np.mean(suppcoefs))
                self.supp_hld.append(dnm)


        #self.recondata = [self.wv,self.wvcoef,self.wvsc,self.wvscoef,self.scwv,self.scwvcoef,self.sc,self.scoef]

        # Scaling Function
        char = np.ones((NY,NX))
        #Afrecon += (np.mean(data) * char)
        

        return Afrecon, self.coefs, self.norms


    def besov_norm(self,data,mNjx,mNjy):

        NY = data.shape[0]
        NX = data.shape[1]
        Afrecon = np.zeros((NY,NX))
        self.coefs = []
        self.norms = []
        self.supp = []
        self.besov = []
        
        self.coefs_hld = []
        self.supp_hld = []
   
        for jxi in range(mNjx-3,mNjx+1):
            for jyi in range(mNjy-3,mNjy+1):

        #for jxi in range(0,2):
        #    for jyi in range(0,2):                
        
                hxv = self.haar(jxi,NX) 
                hyv = self.haar(jyi,NY)
                nxloc = hxv.shape[0]
                nyloc = hyv.shape[0]
                dnm = (2**(-(jxi + jyi)))**(self.alf + 0.5)

                
                suppcoefs = [] # Compute coefficients for each support size
                for kx in range(nxloc):
                    for ky in range(nyloc):                    
                        
                        wvtens = np.kron(hxv[kx,:].reshape(1,NX),hyv[ky,:].reshape(NY,1))
                        acoef = np.abs(np.sum(data * wvtens))
                        #dnm = 2**(-(jxi + jyi))*(self.alf + 0.5)
                        self.norms.append(acoef/dnm)
                        Afrecon += np.sum(data * wvtens) * wvtens
                        self.coefs.append(acoef)
                        self.supp.append(dnm)
                        
                        suppcoefs.append(acoef)

                        # Besov Norm

                        num = acoef**self.p
                        dnm = (2**(-(jxi + jyi)))**((self.alf + 0.5 - (1/self.p))*self.p)
                        self.besov.append(num/dnm) 
                        
                self.coefs_hld.append(np.mean(suppcoefs))
                self.supp_hld.append(dnm)
                

        # Scaling Function
        char = np.ones((NY,NX))
        Afrecon += (np.mean(data) * char)
        

        return Afrecon, self.coefs, (np.sum(self.besov))**(1/self.p)


    def compute_holder_norm(self,pararesid,appresid,odata):

        # Need to use finest scales available
        mXs = int(math.log2(pararesid.shape[0])-1) 
        mYs = int(math.log2(pararesid.shape[1])-1)
        #mXs = 2
        #mYs = 2

        # Compute approximation
        para_resid_wave, para_resid_coefs, para_resid_norms = self.twave_approx(pararesid,mXs,mYs)
        app_resid_wave, app_resid_coefs, app_resid_norms = self.twave_approx(appresid,mXs,mYs)
        f_resid, f_coefs, f_norms = self.twave_approx(odata,mXs,mYs)

        # Compute decomposition
        self.decomp_results = {'paracoeffs': [para_resid_wave, para_resid_coefs, para_resid_norms],'twavecoeffs':[app_resid_wave, app_resid_coefs, app_resid_norms],'datacoeffs':[f_resid, f_coefs, f_norms]}

        return
    
    
    def fast_tpa(self,jeps):
        
        "Generate tensor paraproduct approximation"
        
        "jeps = scale associated with precision we wish to approximate matrix to"
    
        # 1) Compute coefficient matrices
        mWxWy = np.zeros((self.f.shape[0],self.f.shape[1]))
        mWxVy = np.zeros((self.f.shape[0],self.f.shape[1]))
        mVxWy = np.zeros((self.f.shape[0],self.f.shape[1]))
        mVxVy = np.zeros((self.f.shape[0],self.f.shape[1]))

        # 2) Compute all scales at precision epsilon 
        for jx in range(3,jeps):
            jy = jeps - jx
            
            if jeps - jx < 3:
                break
            WxWy, WxVy, VxWy, VxVy = self.genconvs(jx,jy)
            mWxWy += WxWy
            mWxVy += WxVy
            mVxWy += VxWy
            mVxVy += VxVy   
            

        # 3) Compute approximation
    
        # Potential kernel
        if self.kernel == 'varpot':
            termone = -(self.deg + 1)*self.coeffn*self.eps**(-1)*(VxVy**(-(self.deg+2)))
            termtwo = (self.coeffn*(VxVy**(-(self.deg+1)))*self.eps**(-1)) - (self.coeffzro*self.eps**(-1))
            Ap = termone/ (termtwo + 1e-10)
            Adp = ((self.eps - 1)*(self.deg**2 + 2*self.deg + 2)*VxVy**(-2)) + (self.deg*VxVy**(-2)) - (self.coeffzro*(self.coeffn**(-1))*(VxVy**(self.deg + 1)))

        if self.kernel == 'conpot':
            Ap = mVxVy**(-1)
            Adp = -mVxVy**(-2)
            
     
        # Cauchy kernel 
        if self.kernel == 'cauchy':
            Ap = -0.5*mVxVy**(-1.5)
            Adp = 0.75*mVxVy**(-2.5)

        
        self.tpa = (Ap*mWxWy) + (Adp*mWxVy*mVxWy)
        self.jeps = jeps
        

        return self.tpa
    
    
    def tpa_matvec_reg(self, f):
        
        """
        tpa matrix-vector multiplication no coefficient threshold
        g = input function 
        """
        
        # precision of test function is average rectangle side length
        vres = int(self.jeps/2)
        #vres = self.jeps
        N = f.shape[0]
        
        # Generate scaling vectors and project
        sclvecs = self.scl(vres,N)
        avgf = np.zeros(N)
        delt = int(N/(2**vres))
        k=0
        for i in range(sclvecs.shape[0]):
            avgf[k:k+delt] = np.sum(f*sclvecs[i])*np.ones(delt)
            k = k + delt
            

        # Complete matrix-vector multiplication
        og = self.Af @ f
        appg = self.tpa @ avgf
        self.og = og/np.max(np.abs(og))
        self.appg = appg/np.max(np.abs(appg))
   
        return self.og, self.appg

        

        
        
     