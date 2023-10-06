function [A,An] = selftuning2(X,k)

%% load Data
% X = getData(data_name);

%% construct affinity matrix A

[n,~]=size(X);
% ȫ����ŷʽ�������dis_all
dis_all = L2_distance_1(X',X'); % ƽ������
dis_all(find(dis_all<0)) = 0;
dis_all = sqrt(dis_all);

% ʹ��KNN����ŷʽ��������� '��k��'��sigma������n�������ĵ�k��������sigma(n,1)
[dumb,idx] = sort(dis_all, 2); % sort each row
index=idx(:,k+1);

sigma = zeros(n,1);
for i=1:n
    sigma(i)=sqrt(sum((X(i,:)-X(index(i),:)).^2));
end

%����selftuning�µ�affinity����A
A=zeros(n,n);
for i = 1:n                   
    for j=1:n
        if i ~= j
            A(i,j) = exp((-sum((X(i,:)-X(j,:)).^2))/(sigma(i)*sigma(j)));
        end
    end
end

D = diag(sum(A,2));
Dn1 = D^(-0.5);
An = Dn1*A*Dn1; %ǰc�����,An��������˹����
clear Dn;
An = max(An,An');


function d = L2_distance_1(a,b)
% compute squared Euclidean distance
% ||A-B||^2 = ||A||^2 + ||B||^2 - 2*A'*B
% a,b: two matrices. each column is a data
% d:   distance matrix of a and b

if (size(a,1) == 1)
  a = [a; zeros(1,size(a,2))]; 
  b = [b; zeros(1,size(b,2))]; 
end

aa=sum(a.*a); bb=sum(b.*b); ab=a'*b; 
d = repmat(aa',[1 size(bb,2)]) + repmat(bb,[size(aa,2) 1]) - 2*ab;

d = real(d);
d = max(d,0);
