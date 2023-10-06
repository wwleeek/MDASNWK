  function  [sd,sl] = integratedsimilarity2(lncSim,disSim,id,il,gamma)   
 
WD=zeros(1443);
index = find(0 ~= id);
WD(index) = gamma;
sd = WD.*id+(1-WD).*disSim;

WL=zeros(337);
index2 = find(0 ~= il);
WL(index2) = gamma;
sl = WL.*il+(1-WL).*lncSim;
  end