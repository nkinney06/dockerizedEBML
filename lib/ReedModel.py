import assimulo
import numpy as np
import matplotlib.pyplot as plt
import math
from modelbase.ode import Model, LabelModel, LinearLabelModel, Simulator, mca
from modelbase.ode import ratelaws as rl
from modelbase.ode import ratefunctions as rf

def initialVals():
    """return the initial conditions for the reed model"""
    return [19.1432773637, 81.1684566963, 0.942750394172, 12.6996048211, 0.484328542817, 185.503785439, 60.4330872703, 221.101111779, 3.40907070478, 4.49653356534, 0.506278119133, 0.278602708139, 0.0385952337473, 6590.56824161, 61.3019611793, 13.088818643, 4.61991966766, 194.96740946, 9.80842470037, 3219.39793574, 924.429820217, 562.83437727, 36.8825861752, 0.707382712262, 1.12248362562, 15.908798935, 1.66610924423, 1.54929073349, 55.8201166666, 20.9958010884, 49.1868215838, 2040.43402533, 9.16245914629, 2114.8711991,0,30.0,150.0]

def compounds():
    """return the list of compounds in the reed model"""
    return ['SAH', 'SAM', 'aic', 'bGSH', 'bGSSG', 'bcys', 'bglut', 'bgly', 
        'c10fTHF', 'c5mTHF', 'cCH2THF', 'cCHTHF', 'cDHF', 'cGSH', 'cGSSG', 
        'cHCOOH', 'cTHF', 'ccys', 'cglc', 'cglut', 'cgly', 'cser', 'cyt', 
        'dmg', 'hcy', 'm10fTHF', 'mCH2THF', 'mCHTHF', 'mHCOOH', 'mTHF', 'met', 
        'mgly', 'msarc', 'mser','t','bmet','bser']

def parameters(vmGPX,H2O2):
    """return the list of parameters in the reed model"""
    return { 'BET':50.0, 'DUMP':20.0, 'GARP':10.0, 'H2O2':H2O2, 'H2O2ss':0.01, 
        'HCHO':500.0, 'KibhmtH2O2':0.01, 'KmahSAH':6.5, 'Kmahhcy':150.0, 
        'Kmbhmtbet':100.0, 'Kmbhmthcy':12.0, 'KmcARTaic':100.0, 'KmcARTc10fTHF':5.9, 
        'KmcDHFRcDHF':0.5, 'KmcDHFRcNADPH':4.0, 'KmcFTScHCOOH':43.0, 'KmcFTScTHF':3.0, 
        'KmcMTCHc10fTHF':100.0, 'KmcMTCHcCHTHF':250.0, 'KmcMTDcCH2THF':2.0, 
        'KmcMTDcCHTHF':10.0, 'KmcMTHFRcCH2THF':50.0, 'KmcMTHFRcNADPH':16.0, 
        'KmcPGTGARP':520.0, 'KmcPGTc10fTHF':4.9, 'KmcSHMTcCH2THF':3200.0, 'KmcSHMTcTHF':50.0, 
        'KmcSHMTcgly':10000.0, 'KmcSHMTcser':600.0, 'KmcTSDUMP':6.3, 'KmcTScCH2THF':14.0, 
        'Kmcbshcy':1000.0, 'Kmcbsserine':2000.0, 'Kmcdoccys':3100.0, 'KmmDMGDdmg':50.0, 
        'KmmDMGDmTHF':50.0, 'KmmFTSm10fTHF':22.0, 'KmmFTSmHCOOH':43.0, 'KmmFTSmTHF':3.0, 
        'KmmGDCmTHF':50.0, 'KmmGDCmgly':3400.0, 'KmmMTCHm10fTHF':100.0, 'KmmMTCHmCHTHF':250.0, 
        'KmmMTDmCH2THF':2.0, 'KmmMTDmCHTHF':10.0, 'KmmSDHmTHF':50.0, 'KmmSDHsarc':320.0, 
        'KmmSHMTmCH2THF':3200.0, 'KmmSHMTmTHF':50.0, 'KmmSHMTmgly':10000.0, 'KmmSHMTmser':600.0, 
        'KmmethSAH':1.4, 'KmmethSAM':1.4, 'Kmsermser':5700.0, 'Vmbhmt':2160.0, 'VmcART':55000.0, 
        'VmcDHFR':2000.0, 'VmcFTD':500.0, 'VmcFTS':3900.0, 'VmcMTHFR':6000.0, 'VmcPGT':24300.0, 
        'VmcTS':5000.0, 'Vmcbs':420000.0, 'Vmcdo':1327.15, 'Vmfah':320.0, 'VmfcMTCH':500000.0, 
        'VmfcMTD':80000.0, 'VmfcSHMT':5200.0, 'VmfmFTS':2000.0, 'VmfmMTCH':790000.0, 
        'VmfmMTD':180000.0, 'VmfmSHMT':11440.0, 'VmmDMGD':15000.0, 'VmmFTD':1050.0, 
        'VmmGDC':15000.0, 'VmmSDH':15000.0, 'Vmmeth':180.0, 'Vmrah':755.0, 'VmrcMTCH':20000.0, 
        'VmrcMTD':600000.0, 'VmrcSHMT':15000000.0, 'VmrmFTS':6300.0, 'VmrmMTCH':20000.0, 
        'VmrmMTD':594000.0, 'VmrmSHMT':30000000.0, 'Vmser':10000.0, 'cNADPH':50.0, 
        'k0':0.0, 'kaGSSGh':0.01, 'kaGSSGl':0.01, 'kaH2O2':0.035, 'kagcl':0.01, 
        'kbser':150.0, 'kcgly':1.0, 'kcysin':70.0, 'kcysinbcys':2100.0, 'keqgcl':5597.0, 
        'keqgs':5600.0, 'kfcNE':0.03, 'kfmNE':0.03, 'kglutin':1.0, 'kgly':5700.0, 
        'kglyin':630.0, 'khcooh':100.0, 'kiMATiGSSG':2140.0, 'kiMATiiiGSSG':4030.0, 
        'kiMSH2O2':0.01, 'kmGNMTSAH':18.0, 'kmGNMTcgly':130.0, 'kmGPXH2O2':0.01, 'kmGPXgsh':1330.0, 
        'kmGSSGh':1250.0, 'kmGSSGl':7110.0, 'kmMATiSAM':50.0, 'kmMATiiiSAM':129600.0, 
        'kmMATiiimet':300.0, 'kmMATimet':41.0, 'kmMShcy':1.0, 'kmMSmTH4':25.0, 'kmbglut':300.0, 
        'kmbgly':150.0, 'kmcFTDc10fTHF':20.0, 'kmctglcyt':500.0, 'kmetin':30.0, 
        'kmetincmethionine':1.0, 'kmgclccys':100.0, 'kmgclglc':300.0, 'kmgclglut':1900.0, 
        'kmgclgsh':8200.0, 'kmgrGSSG':107.0, 'kmgrNADPH':10.4, 'kmgscglc':22.0, 
        'kmgscgly':300.0, 'kmgshe':3300.0, 'kmgshout':240.0, 'kmgshouth':150.0, 'kmgshoutl':3000.0, 
        'kmmFTDm10fTHF':20.0, 'kmmetinoutmethionine':150.0, 'kpcgsh':30.0, 'krcNE':22.0, 
        'krmNE':20.0, 'krserin':1.0, 'kserin':150.0, 'vmGPX':vmGPX, 'vmGSSGh':40.0, 
        'vmGSSGl':4025.0, 'vmMATi':650.0, 'vmMATiii':220.0, 'vmMS':500.0, 'vmctgl':1500.0, 
        'vmcysin':14950.0, 'vmfgly':10000.0, 'vmgcl':3600.0, 'vmglutin':28000.0, 'vmglyin':4600.0, 
        'vmgr':892.5, 'vmgs':5400.0, 'vmgshe':304.0, 'vmgshout':1000.0, 'vmgshouth':150.0, 
        'vmgshoutl':1100.0, 'vmmetin':913.4, 'vmrgly':10000.0, 'vmserin':2700.0, 'vocysb':70.0, 
        'voglub':273.0, 'voglyb':630.0, 'ext':1.0 }

def t(t):
    return 1

def v1(SAM,met,cGSSG,kiMATiGSSG,vmMATi,kmMATimet):
    return (0.4*(0.23 + 0.8/math.e**(0.0026*SAM))*(66.71 + kiMATiGSSG)*vmMATi*met)/(kmMATimet*(kiMATiGSSG + cGSSG)*(1 + met/kmMATimet))

def v10(ccys):
    return 0.0017499999999999998*ccys**2

def v11(ccys,cglut,cglc,cGSH,H2O2,kagcl,vmgcl,keqgcl,H2O2ss,kmgclccys,kmgclglut,kmgclglc,kmgclgsh):
    return ((H2O2 + kagcl)*vmgcl*(-(cglc/keqgcl) + ccys*cglut))/((H2O2ss + kagcl)*(kmgclccys*kmgclglut + cglc/kmgclglc + kmgclccys*cglut + cGSH/kmgclgsh + kmgclglut*ccys*(1 + cglut/kmgclglut + cGSH/kmgclgsh)))

def v12(cgly,cGSH,cglc,vmgs,keqgs,kmgscglc,kmgscgly,kpcgsh):
    return (vmgs*(cglc*cgly - cGSH/keqgs))/(kmgscglc*kmgscgly + kmgscgly*cglc + kmgscglc*(1 + cglc/kmgscglc)*cgly + cGSH/kpcgsh)

def v13(cGSH,H2O2,vmGPX,kmGPXH2O2,kmGPXgsh):
    return (H2O2*vmGPX*cGSH**2)/((H2O2 + 9*kmGPXH2O2)*(kmGPXgsh + cGSH)**2)

def v14(cGSSG,cNADPH,vmgr,kmgrGSSG,kmgrNADPH):
    return (cNADPH*vmgr*cGSSG)/(kmgrGSSG*kmgrNADPH*(1 + cNADPH/kmgrNADPH + cGSSG/kmgrGSSG + (cNADPH*cGSSG)/(kmgrGSSG*kmgrNADPH)))

def v15(cGSSG,H2O2,kaGSSGl,vmGSSGl,H2O2ss,kmGSSGl):
    return ((H2O2 + kaGSSGl)*vmGSSGl*cGSSG)/((H2O2ss + kaGSSGl)*(kmGSSGl + cGSSG))

def v16(cGSSG,H2O2,kaGSSGh,vmGSSGh,H2O2ss,kmGSSGh):
    return ((H2O2 + kaGSSGh)*vmGSSGh*cGSSG)/((H2O2ss + kaGSSGh)*(kmGSSGh + cGSSG))

def v17(cGSH):
    return 0.002*cGSH

def v18(cGSH,vmgshoutl,kmgshoutl):
    return (vmgshoutl*cGSH**3)/(kmgshoutl**3 + cGSH**3)

def v19(cGSH,vmgshouth,kmgshouth):
    return (vmgshouth*cGSH)/(kmgshouth + cGSH)

def v2(met,SAM,cGSSG,kiMATiiiGSSG,vmMATiii,kmMATiiiSAM,kmMATiiimet):
    return ((66.71 + kiMATiiiGSSG)*vmMATiii*met**1.21*(1 + (7.2*SAM**2)/(kmMATiiiSAM + SAM**2)))/((kiMATiiiGSSG + cGSSG)*(kmMATiiimet + met**1.21))

def v20(cGSSG):
    return 0.1*cGSSG

def v21(bglut,cglut,vmglutin,kmbglut,kglutin):
    return (vmglutin*bglut)/(kmbglut + bglut) - kglutin*cglut

def v22(bcys,vmcysin,kcysinbcys):
    return (vmcysin*bcys)/(kcysinbcys + bcys)

def v23(cser,bser,vmserin,kbser,krserin):
    return (bser*vmserin)/(bser + kbser) - krserin*cser

def v24(bgly,cgly,vmglyin,kmbgly,kcgly):
    return (vmglyin*bgly)/(kmbgly + bgly) - kcgly*cgly

def v25(cgly,mgly,vmrgly,kgly,vmfgly):
    return -((vmrgly*cgly)/(kgly + cgly)) + (vmfgly*mgly)/(3*(kgly + mgly))

def v26(cHCOOH,mHCOOH,khcooh):
    return -(khcooh*cHCOOH) + (khcooh*mHCOOH)/3

def v27(cser,mser,Vmser,Kmsermser):
    return -((Vmser*cser)/(Kmsermser + cser)) + (Vmser*mser)/(3*(Kmsermser + mser))

def v28(cgly,cCH2THF,cser,cTHF,VmrcSHMT,KmcSHMTcCH2THF,KmcSHMTcgly,VmfcSHMT,KmcSHMTcser,KmcSHMTcTHF):
    return -((VmrcSHMT*cCH2THF*cgly)/(KmcSHMTcCH2THF*KmcSHMTcgly*(1 + cCH2THF/KmcSHMTcCH2THF + cgly/KmcSHMTcgly + (cCH2THF*cgly)/(KmcSHMTcCH2THF*KmcSHMTcgly)))) + (VmfcSHMT*cser*cTHF)/(KmcSHMTcser*KmcSHMTcTHF*(1 + cser/KmcSHMTcser + cTHF/KmcSHMTcTHF + (cser*cTHF)/(KmcSHMTcser*KmcSHMTcTHF)))

def v29(cDHF,cNADPH,VmcDHFR,KmcDHFRcDHF,KmcDHFRcNADPH):
    return (cNADPH*VmcDHFR*cDHF)/(KmcDHFRcDHF*KmcDHFRcNADPH*(1 + cNADPH/KmcDHFRcNADPH + cDHF/KmcDHFRcDHF + (cNADPH*cDHF)/(KmcDHFRcDHF*KmcDHFRcNADPH)))

def v3(SAM,SAH,Vmmeth,KmmethSAM,KmmethSAH):
    return (Vmmeth*SAM)/(KmmethSAM*(1 + SAH/KmmethSAH) + SAM)

def v30(cHCOOH,cTHF,VmcFTS,KmcFTScHCOOH,KmcFTScTHF):
    return (VmcFTS*cHCOOH*cTHF)/(KmcFTScHCOOH*KmcFTScTHF*(1 + cHCOOH/KmcFTScHCOOH + cTHF/KmcFTScTHF + (cHCOOH*cTHF)/(KmcFTScHCOOH*KmcFTScTHF)))

def v31(c10fTHF,VmcFTD,kmcFTDc10fTHF):
    return (VmcFTD*c10fTHF)/(kmcFTDc10fTHF*(1 + c10fTHF/kmcFTDc10fTHF))

def v32(c10fTHF,GARP,VmcPGT,KmcPGTc10fTHF,KmcPGTGARP):
    return (GARP*VmcPGT*c10fTHF)/(KmcPGTc10fTHF*KmcPGTGARP*(1 + GARP/KmcPGTGARP + c10fTHF/KmcPGTc10fTHF + (GARP*c10fTHF)/(KmcPGTc10fTHF*KmcPGTGARP)))

def v33(aic,c10fTHF,VmcART,KmcARTaic,KmcARTc10fTHF):
    return (VmcART*aic*c10fTHF)/(KmcARTaic*KmcARTc10fTHF*(1 + aic/KmcARTaic + c10fTHF/KmcARTc10fTHF + (aic*c10fTHF)/(KmcARTaic*KmcARTc10fTHF)))

def v34(cCH2THF,cTHF,krcNE,HCHO,kfcNE):
    return -(krcNE*cCH2THF) + HCHO*kfcNE*cTHF

def v35(c10fTHF,cCHTHF,VmrcMTCH,KmcMTCHc10fTHF,VmfcMTCH,KmcMTCHcCHTHF):
    return -((VmrcMTCH*c10fTHF)/(KmcMTCHc10fTHF*(1 + c10fTHF/KmcMTCHc10fTHF))) + (VmfcMTCH*cCHTHF)/(KmcMTCHcCHTHF*(1 + cCHTHF/KmcMTCHcCHTHF))

def v36(cCH2THF,DUMP,VmcTS,KmcTScCH2THF,KmcTSDUMP):
    return (DUMP*VmcTS*cCH2THF)/(KmcTScCH2THF*KmcTSDUMP*(1 + DUMP/KmcTSDUMP + cCH2THF/KmcTScCH2THF + (DUMP*cCH2THF)/(KmcTScCH2THF*KmcTSDUMP)))

def v37(cCH2THF,cCHTHF,VmfcMTD,KmcMTDcCH2THF,VmrcMTD,KmcMTDcCHTHF):
    return (VmfcMTD*cCH2THF)/(KmcMTDcCH2THF*(1 + cCH2THF/KmcMTDcCH2THF)) - (VmrcMTD*cCHTHF)/(KmcMTDcCHTHF*(1 + cCHTHF/KmcMTDcCHTHF))

def v38(cCH2THF,SAM,SAH,cNADPH,VmcMTHFR,KmcMTHFRcCH2THF,KmcMTHFRcNADPH):
    return (63.72*cNADPH*VmcMTHFR*cCH2THF)/(KmcMTHFRcCH2THF*KmcMTHFRcNADPH*(1 + cNADPH/KmcMTHFRcNADPH + cCH2THF/KmcMTHFRcCH2THF + (cNADPH*cCH2THF)/(KmcMTHFRcCH2THF*KmcMTHFRcNADPH))*(10 + (0 if (SAM[-1] < SAH[-1]) else (SAM[-1]-SAH[-1]))))

def v39(mCH2THF,mgly,mser,mTHF,VmrmSHMT,KmmSHMTmCH2THF,KmmSHMTmgly,VmfmSHMT,KmmSHMTmser,KmmSHMTmTHF):
    return -((VmrmSHMT*mCH2THF*mgly)/(KmmSHMTmCH2THF*KmmSHMTmgly*(1 + mCH2THF/KmmSHMTmCH2THF + mgly/KmmSHMTmgly + (mCH2THF*mgly)/(KmmSHMTmCH2THF*KmmSHMTmgly)))) + (VmfmSHMT*mser*mTHF)/(KmmSHMTmser*KmmSHMTmTHF*(1 + mser/KmmSHMTmser + mTHF/KmmSHMTmTHF + (mser*mTHF)/(KmmSHMTmser*KmmSHMTmTHF)))

def v4(cgly,SAM,c5mTHF,SAH,kmGNMTcgly,kmGNMTSAH):
    return (1248.*cgly*SAM)/((0.35 + c5mTHF)*(kmGNMTcgly + cgly)*(1 + SAH/kmGNMTSAH)*(63 + SAM))

def v40(m10fTHF,mHCOOH,mTHF,VmrmFTS,KmmFTSm10fTHF,VmfmFTS,KmmFTSmHCOOH,KmmFTSmTHF):
    return -((VmrmFTS*m10fTHF)/(KmmFTSm10fTHF*(1 + m10fTHF/KmmFTSm10fTHF))) + (VmfmFTS*mHCOOH*mTHF)/(KmmFTSmHCOOH*KmmFTSmTHF*(1 + mHCOOH/KmmFTSmHCOOH + mTHF/KmmFTSmTHF + (mHCOOH*mTHF)/(KmmFTSmHCOOH*KmmFTSmTHF)))

def v41(m10fTHF,VmmFTD,kmmFTDm10fTHF):
    return (VmmFTD*m10fTHF)/(kmmFTDm10fTHF*(1 + m10fTHF/kmmFTDm10fTHF))

def v42(m10fTHF,mCHTHF,VmrmMTCH,KmmMTCHm10fTHF,VmfmMTCH,KmmMTCHmCHTHF):
    return -((VmrmMTCH*m10fTHF)/(KmmMTCHm10fTHF*(1 + m10fTHF/KmmMTCHm10fTHF))) + (VmfmMTCH*mCHTHF)/(KmmMTCHmCHTHF*(1 + mCHTHF/KmmMTCHmCHTHF))

def v43(mCH2THF,mCHTHF,VmfmMTD,KmmMTDmCH2THF,VmrmMTD,KmmMTDmCHTHF):
    return (VmfmMTD*mCH2THF)/(KmmMTDmCH2THF*(1 + mCH2THF/KmmMTDmCH2THF)) - (VmrmMTD*mCHTHF)/(KmmMTDmCHTHF*(1 + mCHTHF/KmmMTDmCHTHF))

def v44(mgly,mTHF,VmmGDC,KmmGDCmgly,KmmGDCmTHF):
    return (VmmGDC*mgly*mTHF)/(KmmGDCmgly*KmmGDCmTHF*(1 + mgly/KmmGDCmgly + mTHF/KmmGDCmTHF + (mgly*mTHF)/(KmmGDCmgly*KmmGDCmTHF)))

def v45(msarc,mTHF,VmmSDH,KmmSDHmTHF,KmmSDHsarc):
    return (VmmSDH*msarc*mTHF)/(KmmSDHmTHF*KmmSDHsarc*(1 + msarc/KmmSDHsarc + mTHF/KmmSDHmTHF + (msarc*mTHF)/(KmmSDHmTHF*KmmSDHsarc)))

def v46(dmg,mTHF,VmmDMGD,KmmDMGDdmg,KmmDMGDmTHF):
    return (VmmDMGD*dmg*mTHF)/(KmmDMGDdmg*KmmDMGDmTHF*(1 + dmg/KmmDMGDdmg + mTHF/KmmDMGDmTHF + (dmg*mTHF)/(KmmDMGDdmg*KmmDMGDmTHF)))

def v47(mCH2THF,mTHF,krmNE,HCHO,kfmNE):
    return -(krmNE*mCH2THF) + HCHO*kfmNE*mTHF

def v48(met,bmet,vmmetin,kmmetinoutmethionine,kmetincmethionine):
    return (bmet*vmmetin)/(bmet + kmmetinoutmethionine) - kmetincmethionine*met

def v49(bGSH,bGSSG,vocysb,k0):
    return vocysb + k0*bGSH*bGSSG

def v5(hcy,SAH,Vmrah,Kmahhcy,Vmfah,KmahSAH):
    return (-6*Vmrah*hcy)/(Kmahhcy*(1 + hcy/Kmahhcy)) + (Vmfah*SAH)/(KmahSAH*(1 + SAH/KmahSAH))

def v50(bGSH,bGSSG,voglub,k0):
    return voglub + k0*bGSH*bGSSG

def v51(bGSH,bGSSG,voglyb,k0):
    return voglyb + k0*bGSH*bGSSG

def v52(cser):
    return 1.2*cser

def v53(cglut):
    return 0.07*cglut

def v54(bGSH):
    return 90*bGSH

def v55(bGSSG):
    return (135*bGSSG)/2

def v56(bgly):
    return 0.1*bgly

def v57(bcys):
    return 0.35*bcys

def v58(bglut):
    return 0.1*bglut

def v59(bGSH):
    return 0.7*bGSH

def v6(hcy,SAM,SAH,BET,H2O2ss,KibhmtH2O2,Vmbhmt,H2O2,Kmbhmtbet,Kmbhmthcy):
    return (1.2404063158300271*BET*(H2O2ss + KibhmtH2O2)*Vmbhmt*hcy)/(math.e**(0.0021*(SAH + SAM))*(H2O2 + KibhmtH2O2)*(BET + Kmbhmtbet)*Kmbhmthcy*(1 + hcy/Kmbhmthcy))

def v60(bGSSG):
    return 7.5*bGSSG

def v61(bmet):
    return bmet
    
def v62(bser):
    return bser
    
def v7(c5mTHF,hcy,H2O2ss,kiMSH2O2,vmMS,H2O2,kmMShcy,kmMSmTH4):
    return ((H2O2ss + kiMSH2O2)*vmMS*c5mTHF*hcy)/((H2O2 + kiMSH2O2)*kmMShcy*kmMSmTH4*(1 + c5mTHF/kmMSmTH4)*(1 + hcy/kmMShcy))

def v8(cser,hcy,SAM,SAH,H2O2,kaH2O2,Vmcbs,H2O2ss,Kmcbshcy,Kmcbsserine):
    return (1.0855130604524792*(H2O2 + kaH2O2)*Vmcbs*cser*hcy)/((H2O2ss + kaH2O2)*Kmcbshcy*(Kmcbsserine + cser)*(1 + hcy/Kmcbshcy)*(1 + 900/(SAH + SAM)**2))

def v9(cyt,vmctgl,kmctglcyt):
    return (vmctgl*cyt)/(kmctglcyt*(1 + cyt/kmctglcyt))

def getModel(cmpds,p):
    """return the reed model"""
    m = Model(p)
    m.add_compounds(cmpds)
    m.add_rate(rate_name="v1",  function=v1,  substrates=['SAM','met','cGSSG'], parameters=['kiMATiGSSG','vmMATi','kmMATimet'])
    m.add_rate(rate_name="v2",  function=v2,  substrates=['met','SAM','cGSSG'], parameters=['kiMATiiiGSSG','vmMATiii','kmMATiiiSAM','kmMATiiimet'])
    m.add_rate(rate_name="v3",  function=v3,  substrates=['SAM','SAH'], parameters=['Vmmeth','KmmethSAM','KmmethSAH'])
    m.add_rate(rate_name="v4",  function=v4,  substrates=['cgly','SAM','c5mTHF','SAH'], parameters=['kmGNMTcgly','kmGNMTSAH'])
    m.add_rate(rate_name="v5",  function=v5,  substrates=['hcy','SAH'], parameters=['Vmrah','Kmahhcy','Vmfah','KmahSAH'])
    m.add_rate(rate_name="v6",  function=v6,  substrates=['hcy','SAM','SAH'], parameters=['BET','H2O2ss','KibhmtH2O2','Vmbhmt','H2O2','Kmbhmtbet','Kmbhmthcy'])
    m.add_rate(rate_name="v7",  function=v7,  substrates=['c5mTHF','hcy'], parameters=['H2O2ss','kiMSH2O2','vmMS','H2O2','kmMShcy','kmMSmTH4'])
    m.add_rate(rate_name="v8",  function=v8,  substrates=['cser','hcy','SAM','SAH'], parameters=['H2O2','kaH2O2','Vmcbs','H2O2ss','Kmcbshcy','Kmcbsserine'])
    m.add_rate(rate_name="v9",  function=v9,  substrates=['cyt'], parameters=['vmctgl','kmctglcyt'])
    m.add_rate(rate_name="v10", function=v10, substrates=['ccys'], parameters=[])
    m.add_rate(rate_name="v11", function=v11, substrates=['ccys','cglut','cglc','cGSH'], parameters=['H2O2','kagcl','vmgcl','keqgcl','H2O2ss','kmgclccys','kmgclglut','kmgclglc','kmgclgsh'])
    m.add_rate(rate_name="v12", function=v12, substrates=['cgly','cGSH','cglc'], parameters=['vmgs','keqgs','kmgscglc','kmgscgly','kpcgsh'])
    m.add_rate(rate_name="v13", function=v13, substrates=['cGSH'], parameters=['H2O2','vmGPX','kmGPXH2O2','kmGPXgsh'])
    m.add_rate(rate_name="v14", function=v14, substrates=['cGSSG'], parameters=['cNADPH','vmgr','kmgrGSSG','kmgrNADPH'])
    m.add_rate(rate_name="v15", function=v15, substrates=['cGSSG'], parameters=['H2O2','kaGSSGl','vmGSSGl','H2O2ss','kmGSSGl'])
    m.add_rate(rate_name="v16", function=v16, substrates=['cGSSG'], parameters=['H2O2','kaGSSGh','vmGSSGh','H2O2ss','kmGSSGh'])
    m.add_rate(rate_name="v17", function=v17, substrates=['cGSH'], parameters=[])
    m.add_rate(rate_name="v18", function=v18, substrates=['cGSH'], parameters=['vmgshoutl','kmgshoutl'])
    m.add_rate(rate_name="v19", function=v19, substrates=['cGSH'], parameters=['vmgshouth','kmgshouth'])
    m.add_rate(rate_name="v20", function=v20, substrates=['cGSSG'], parameters=[])
    m.add_rate(rate_name="v21", function=v21, substrates=['bglut','cglut'], parameters=['vmglutin','kmbglut','kglutin'])
    m.add_rate(rate_name="v22", function=v22, substrates=['bcys'], parameters=['vmcysin','kcysinbcys'])
    m.add_rate(rate_name="v23", function=v23, substrates=['cser','bser'], parameters=['vmserin','kbser','krserin'])
    m.add_rate(rate_name="v24", function=v24, substrates=['bgly','cgly'], parameters=['vmglyin','kmbgly','kcgly'])
    m.add_rate(rate_name="v25", function=v25, substrates=['cgly','mgly'], parameters=['vmrgly','kgly','vmfgly'])
    m.add_rate(rate_name="v26", function=v26, substrates=['cHCOOH','mHCOOH'], parameters=['khcooh'])
    m.add_rate(rate_name="v27", function=v27, substrates=['cser','mser'], parameters=['Vmser','Kmsermser'])
    m.add_rate(rate_name="v28", function=v28, substrates=['cgly','cCH2THF','cser','cTHF'], parameters=['VmrcSHMT','KmcSHMTcCH2THF','KmcSHMTcgly','VmfcSHMT','KmcSHMTcser','KmcSHMTcTHF'])
    m.add_rate(rate_name="v29", function=v29, substrates=['cDHF'], parameters=['cNADPH','VmcDHFR','KmcDHFRcDHF','KmcDHFRcNADPH'])
    m.add_rate(rate_name="v30", function=v30, substrates=['cHCOOH','cTHF'], parameters=['VmcFTS','KmcFTScHCOOH','KmcFTScTHF'])
    m.add_rate(rate_name="v31", function=v31, substrates=['c10fTHF'], parameters=['VmcFTD','kmcFTDc10fTHF'])
    m.add_rate(rate_name="v32", function=v32, substrates=['c10fTHF'], parameters=['GARP','VmcPGT','KmcPGTc10fTHF','KmcPGTGARP'])
    m.add_rate(rate_name="v33", function=v33, substrates=['aic','c10fTHF'], parameters=['VmcART','KmcARTaic','KmcARTc10fTHF'])
    m.add_rate(rate_name="v34", function=v34, substrates=['cCH2THF','cTHF'], parameters=['krcNE','HCHO','kfcNE'])
    m.add_rate(rate_name="v35", function=v35, substrates=['c10fTHF','cCHTHF'], parameters=['VmrcMTCH','KmcMTCHc10fTHF','VmfcMTCH','KmcMTCHcCHTHF'])
    m.add_rate(rate_name="v36", function=v36, substrates=['cCH2THF'], parameters=['DUMP','VmcTS','KmcTScCH2THF','KmcTSDUMP'])
    m.add_rate(rate_name="v37", function=v37, substrates=['cCH2THF','cCHTHF'], parameters=['VmfcMTD','KmcMTDcCH2THF','VmrcMTD','KmcMTDcCHTHF'])
    m.add_rate(rate_name="v38", function=v38, substrates=['cCH2THF','SAM','SAH'], parameters=['cNADPH','VmcMTHFR','KmcMTHFRcCH2THF','KmcMTHFRcNADPH'])
    m.add_rate(rate_name="v39", function=v39, substrates=['mCH2THF','mgly','mser','mTHF'], parameters=['VmrmSHMT','KmmSHMTmCH2THF','KmmSHMTmgly','VmfmSHMT','KmmSHMTmser','KmmSHMTmTHF'])
    m.add_rate(rate_name="v40", function=v40, substrates=['m10fTHF','mHCOOH','mTHF'], parameters=['VmrmFTS','KmmFTSm10fTHF','VmfmFTS','KmmFTSmHCOOH','KmmFTSmTHF'])
    m.add_rate(rate_name="v41", function=v41, substrates=['m10fTHF'], parameters=['VmmFTD','kmmFTDm10fTHF'])
    m.add_rate(rate_name="v42", function=v42, substrates=['m10fTHF','mCHTHF'], parameters=['VmrmMTCH','KmmMTCHm10fTHF','VmfmMTCH','KmmMTCHmCHTHF'])
    m.add_rate(rate_name="v43", function=v43, substrates=['mCH2THF','mCHTHF'], parameters=['VmfmMTD','KmmMTDmCH2THF','VmrmMTD','KmmMTDmCHTHF'])
    m.add_rate(rate_name="v44", function=v44, substrates=['mgly','mTHF'], parameters=['VmmGDC','KmmGDCmgly','KmmGDCmTHF'])
    m.add_rate(rate_name="v45", function=v45, substrates=['msarc','mTHF'], parameters=['VmmSDH','KmmSDHmTHF','KmmSDHsarc'])
    m.add_rate(rate_name="v46", function=v46, substrates=['dmg','mTHF'], parameters=['VmmDMGD','KmmDMGDdmg','KmmDMGDmTHF'])
    m.add_rate(rate_name="v47", function=v47, substrates=['mCH2THF','mTHF'], parameters=['krmNE','HCHO','kfmNE'])
    m.add_rate(rate_name="v48", function=v48, substrates=['met','bmet'], parameters=['vmmetin','kmmetinoutmethionine','kmetincmethionine'])
    m.add_rate(rate_name="v49", function=v49, substrates=['bGSH','bGSSG'], parameters=['vocysb','k0'])
    m.add_rate(rate_name="v50", function=v50, substrates=['bGSH','bGSSG'], parameters=['voglub','k0'])
    m.add_rate(rate_name="v51", function=v51, substrates=['bGSH','bGSSG'], parameters=['voglyb','k0'])
    m.add_rate(rate_name="v52", function=v52, substrates=['cser'], parameters=[])
    m.add_rate(rate_name="v53", function=v53, substrates=['cglut'], parameters=[])
    m.add_rate(rate_name="v54", function=v54, substrates=['bGSH'], parameters=[])
    m.add_rate(rate_name="v55", function=v55, substrates=['bGSSG'], parameters=[])
    m.add_rate(rate_name="v56", function=v56, substrates=['bgly'], parameters=[])
    m.add_rate(rate_name="v57", function=v57, substrates=['bcys'], parameters=[])
    m.add_rate(rate_name="v58", function=v58, substrates=['bglut'], parameters=[])
    m.add_rate(rate_name="v59", function=v59, substrates=['bGSH'], parameters=[])
    m.add_rate(rate_name="v60", function=v60, substrates=['bGSSG'], parameters=[])
    m.add_rate(rate_name="v61", function=v61, substrates=['bmet'], parameters=[])
    m.add_rate(rate_name="v62", function=v62, substrates=['bser'], parameters=[])
    m.add_rate(rate_name="t", function=t, substrates=['t'], parameters=[])
    
    m.add_stoichiometries_by_compounds({
        "SAH": { "v3":  1, "v4":  1, "v5":  -1},
        "SAM": { "v1":  1, "v2":  1, "v3":  -1, "v4":  -1},
        "aic": { "v32":  1, "v33":  -1},
        "bGSH": { "v18":  1, "v19":  1, "v59":  -1, "v54":  -1},
        "bGSSG": { "v16":  1, "v15":  1, "v55":  -1, "v60":  -1},
        "bcys": { "v55":  2, "v49":  1, "v54":  1, "v57":  -1, "v22":  -1},
        "bglut": { "v55":  2, "v54":  1, "v50":  1, "v21":  -1, "v58":  -1},
        "bgly": { "v55":  2, "v51":  1, "v54":  1, "v24":  -1, "v56":  -1},
        "c10fTHF": { "v30":  1, "v35":  1, "v32":  -1, "v31":  -1, "v33":  -1},
        "c5mTHF": { "v38":  1, "v7":  -1},
        "cCH2THF": { "v28":  1, "v34":  1, "v36":  -1, "v37":  -1, "v38":  -1},
        "cCHTHF": { "v37":  1, "v35":  -1},
        "cDHF": { "v36":  1, "v29":  -1},
        "cGSH": { "v12":  1, "v14":  2, "v18":  -1, "v13":  -2, "v17":  -1, "v19":  -1},
        "cGSSG": { "v13":  1, "v20":  -1, "v16":  -1, "v15":  -1, "v14":  -1},
        "cHCOOH": { "v26":  1, "v30":  -1},
        "cTHF": { "v32":  1, "v31":  1, "v7":  1, "v29":  1, "v33":  1, "v30":  -1, "v28":  -1, "v34":  -1},
        "ccys": { "v9":  1, "v22":  1, "v10":  -1, "v11":  -1},
        "cglc": { "v11":  1, "v12":  -1},
        "cglut": { "v21":  1, "v53":  -1, "v11":  -1},
        "cgly": { "v24":  1, "v25":  1, "v28":  1, "v12":  -1, "v4":  -1},
        "cser": { "v23":  1, "v27":  1, "v8":  -1, "v52":  -1, "v28":  -1},
        "cyt": { "v8":  1, "v9":  -1},
        "dmg": { "v6":  1, "v46":  -1},
        "hcy": { "v5":  1, "v8":  -1, "v7":  -1, "v6":  -1},
        "m10fTHF": { "v40":  1, "v42":  1, "v41":  -1},
        "mCH2THF": { "v44":  1, "v45":  1, "v47":  1, "v46":  1, "v39":  1, "v43":  -1},
        "mCHTHF": { "v43":  1, "v42":  -1},
        "mHCOOH": { "v40":  -1, "v26":  -3},
        "mTHF": { "v41":  1, "v44":  -1, "v45":  -1, "v40":  -1, "v47":  -1, "v46":  -1, "v39":  -1},
        "met": { "v48":  1, "v7":  1, "v6":  1, "v1":  -1, "v2":  -1},
        "mgly": { "v45":  1, "v39":  1, "v44":  -1, "v25":  -3},
        "msarc": { "v4":  1, "v46":  1, "v45":  -1},
        "mser": { "v27":  -3, "v39":  -1},
        "bmet": { "v61": 0 },
        "bser": { "v62": 0 },
        "t": { "t": 1 }
        })
    return m

def steadyStateDict(toTime,vmGPX,H2O2):
    c = compounds()
    p = parameters(vmGPX,H2O2)
    m = getModel(c,p)
    i = initialVals()
    s = Simulator(m)
    s.initialise(i)
    t, y = s.simulate(toTime)
    fluxes = s.get_fluxes_dict()
    result = s.get_results_dict()
    results = {'flux': {k:fluxes[k][-1] for k in fluxes.keys()}, 'cmpd': {k:result[k][-1] for k in result.keys()}}
    return results

