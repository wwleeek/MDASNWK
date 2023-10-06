%% Load data set
lncSim = load ('DJS.txt');
%lncSim = load ('lncRNAJS770.txt');
interaction = importdata ('D-M_337x1444.txt');
%interaction = importdata ('lncRNA_miRNA770x275.txt');
disSim = load('MJS.txt');
%disSim = load('miRNAJS275.txt');
%% Construct two weight matrices
 WD=zeros(1443);
 %WD=zeros(275);
 index = find(0 ~= disSim);
 WL=zeros(337);
 %WL=zeros(770);
 index1 = find(0 ~= lncSim);
 [nl,nd] = size(interaction);
%% Retain the original correlation matrix and transpose
 interaction = interaction';
 save interaction interaction;
%% Calculate the cosine similarity matrix and integrate
 [id ,il] = cosSim( interaction );                   
 [sd,sl] = integratedsimilarity2(lncSim,disSim,id,il,0.5);   
  
%% Calculated scoring matrix
 [NCP]=MDASNWK(interaction,sd,sl);

%% Five fold cross validation
index_1 = find(1 == interaction);
auc=zeros(1,100);
pp = length(index_1);
   for i = 1 : 5
    i
    indices = crossvalind('Kfold', pp, 5); 
    for j = 1:5 
       
        index_2 = find(j == indices);
        load interaction;
        interaction(index_1(index_2)) = 0;
        [id ,il] = cosSim( interaction);                    
        [sd,sl] = integratedsimilarity2(lncSim,disSim,id,il,0.5); 
        [result]=MDASNWK(interaction,sd,sl);
        NCP(index_1(index_2)) = result(index_1(index_2)); 
    end
    pre_label_score = NCP(:);
    save D_M_337x1444_5kcv_without_WK pre_label_score;
    load interaction;
    label_y = interaction(:);
    auc(i) = roc_1(pre_label_score,label_y,'red');
    %auc(i) = roc_2(pre_label_score,label_y,'red');

   end
   auc_avg = mean(auc);
   std(auc);
 