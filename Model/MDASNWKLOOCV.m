%% Load data set

%[~,disease]=xlsread(['Dataset1\miRNA_265.xlsx']);
%[~,lncRNA]=xlsread(['Dataset1\lncRNA_417.xlsx']);
lncSim = load ('DJS.txt');
interaction = importdata ('D-M_337x1444.txt');
disSim = load('MJS.txt');
KK = 3;
r = 0.5;
%%
interaction = WKNKN( interaction, lncSim, disSim, KK, r );
%% Construct two weight matrices
WD=zeros(1443);
index = find(0 ~= disSim);
WD(index) = 0.5;
WL=zeros(337);
index1 = find(0 ~= lncSim);
WL(index1) = 0.5;
[nl,nd] = size(interaction);

%% Retain the original correlation matrix and transpose
interaction = interaction';
save interaction interaction;
%% Calculate the cosine similarity matrix and integrate
[id ,il] = cosSim( interaction );                    % Return the processed cosine similarity matrix
[sd,sl] = integratedsimilarity2(lncSim,disSim,id,il,0.5);  % Integrated similarity for diseases and miRNAs   

  
%% Calculated scoring matrix
[NCP]=MDASNWK(interaction,sd,sl);

%% result
%allresult(disease,lncRNA,interaction,NCP);


%% Leave a cross-validation
% auc_all = zeros(1,9);
% q = 1;
% for gamma = 0.1:0.1:0.9
%     gamma
% [id ,il] = cosSim( interaction );                   
% [sd,sl] = integratedsimilarity2(lncSim,disSim,id,il,gamma);  
% [NCP]=NCPHLDA(interaction,sd,sl);
index_1 = find(1 == interaction);

%for i = 1:length(index_1)
for i = 1:10
        i
        load interaction;
        interaction(index_1(i))=0;
        [id ,il] = cosSim(interaction);   
        [sd,sl] = integratedsimilarity2(lncSim,disSim,id,il,0.5);   
       %%
        interaction = WKNKN( interaction', lncSim, disSim, KK, r );
        [result]=MDASNWK(interaction',sd,sl);
        NCP(index_1(i)) = result(index_1(i));
        
       
end
% 
%     pre_label_score = NCP(:);
%     label_y = interaction(:);
%     auc = roc_1(pre_label_score,label_y,'red');
%     auc_all(q) = auc;
%     q = q+1;
% end
% x=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9];
% plot(x,auc_all)
% xlabel('¦Á');
% ylabel('AUC');
% grid on
% set(gca,'YLim',[0.7 1]);
% axis([x auc_all],[0 1 0 1])
    pre_label_score = NCP(:);
%     save pre_label_score_NCPHLDA_lncSim_disSim pre_label_score;
%     save pre_label_score_onlylncSimanddisSim pre_label_score;
   save LOOCV pre_label_score;
    load interaction;
    label_y = interaction(:);
    auc = roc_1(pre_label_score,label_y,'blue');
