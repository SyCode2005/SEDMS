%模型6的代码：||B^T-WE^T||+aTr(E^TLE)+beta||W||_{2,1}
%s.t.W^TW=I,E=XW
function [valueArr,alpha1,gamma1,gamma2,beta,score_sort,featureSelectIndex]=featureSelectionFun6(matrixSketch,MF_topK,shrinkLevel,perFeature,maxIter,alpha1,gamma1,gamma2,beta)
[matrixSketchI1,matrixSketchI2]=size(matrixSketch);
%交替更新W和E
converge=0;
W=rand(matrixSketchI2,MF_topK);
P=rand(matrixSketchI1,MF_topK);
%修改
currMatrix=matrixSketch(shrinkLevel:end,:);
[I1,I2]=size(currMatrix);
P2=P(shrinkLevel:end,:);
[L,~,~]=construct_Laplacian(currMatrix);%L的维度：matrixSketchI1*matrixSketchI1

%下面是对L和U2进行维度，扩展的部分填0
L_large=zeros(matrixSketchI1,matrixSketchI1);
L_large(shrinkLevel:end,shrinkLevel:end)=L;
P2_large=zeros(matrixSketchI1,MF_topK);
P2_large(shrinkLevel:end,:)=P2;


%alpha1=10000;%1 谱信息的重要性
%beta=10000;
%gamma1=10^4;%%控制U的正交程度  10^7
%gamma2=10^4;%对应论文的alpha，控制稀疏度

M=alpha1*L_large;%%修改  M=alpha1*L;
M_plus=0.5*(abs(M)+M);
M_minus=0.5*(abs(M)-M);
iter=1;
old_error=0;
tol=1e-10;
valueArr=zeros(1,maxIter);
D=eye(matrixSketchI2,matrixSketchI2);
currError=zeros(1,maxIter);
while (iter<maxIter) && (converge==0)
    %Update W
    %construct D
    if(iter>1)
        D=diag(sparse(0.5./max(sqrt(sum(W.^2,2)),1e-20)));
    end
    upper_tmp=matrixSketch'*P+2*gamma1*W+gamma2*matrixSketch'*eye(matrixSketchI1,MF_topK);
    %upper_tmp=matrixSketch'*E++2*gamma1*W;
    lower_tmp=W*P'*P+0.5*beta*D*W+2*gamma1*W*W'*W;
    W=W.*(upper_tmp./max(lower_tmp,1e-20)).^(0.5);
    
    
    %Update P
    upper_tmp=2*matrixSketch*W+2*M_minus*P2_large;
    lower_tmp=2*P*W'*W+2*M_plus*P2_large+gamma2*eye(matrixSketchI1,MF_topK);
    %lower_tmp=2*E*W'*W+2*M_plus*E;
    P= P.*(upper_tmp./max(lower_tmp,1e-20)).^(0.5);
    
    %Update iter
    

    %Update iter
    train_error=norm(matrixSketch'-W*P','fro')+alpha1*trace(P2'*L*P2);
    %[train_error]=recover_error(matrixSketch',W*E');
    valueArr(1,iter)=abs(train_error-old_error);
    if (valueArr(1,iter)<tol)
        converge=1;
        %disp("old_error="+old_error+",train_error="+train_error+",iter="+iter);
    end
    old_error=train_error;
    iter=iter + 1;
end
%特征选择
score=zeros(matrixSketchI2,1);
for i=1:matrixSketchI2
    score(i,1)=norm(W(i,:),2);
end
[score_sort,score_sort_index]=sort(score,'descend');

num_featureSelect=round(matrixSketchI2*perFeature);%特征选择率
featureSelectIndex=score_sort_index(1:num_featureSelect,:);
featureSelectIndex=sort(featureSelectIndex);

%feaSelection_data=matrixSketch(:,featureSelectIndex);

[~,length]=size(valueArr);
timeArr=zeros(1,length);
for i=1:length
    timeArr(1,i)=i;
end
plot(timeArr,valueArr);
end

function [L,V,S] = construct_Laplacian(X)
n=size(X,1);
S = kernelmatrix(X,1);
V = zeros(size(S));
for i = 1:size(V,1)
    V(i,i) = sum(S(:,i));
end
L = V - S;
end

function K = kernelmatrix(coord, sig)
n=size(coord,1);
K=coord*coord'/sig^2;
d=diag(K);
K=K-ones(n,1)*d'/2;
K=K-d*ones(1,n)/2;
K=exp(K);
end

%涉及穷尽算法的KNN
%DataAll:所有的数据集  number*d
%number：样本数量
%KNNk：寻找每个样本最近的KNNk个样本
function [S,D,L]=conSimilarMatrix(DataAll,number,KNNk)
t=1;
G=zeros(number,number);
%对每个元素计算KNNk个最近元素：
for i=1:number
   element=DataAll(i,:); 
   %Idx是element对应的KNNk个最近元素的下标
   Idx=knnsearch(DataAll,element,'K',KNNk);
   for j=1:KNNk
       G(i,Idx(1,j))=1;
   end
end
S=zeros(number,number);%相似矩阵，对角元素都是1，不知道对不对
for i=1:number
    data1=DataAll(i,:);
    for j=1:number
        data2=DataAll(j,:);
        if(i==j)
            S(i,j)=1;
        elseif(G(i,j)==1)
            S(i,j)=exp(-norm(data1-data2,'fro')^2/t);
        end
    end
end 
D=diag(S*ones(number,1));
L=D-S;
% D1=zeros(number,number);
% for i=1:number
%     D1(i,i)=sum(S(i,:));
% end
% L1=D1-S;
end


function [train_error]=recover_error(X,recoverX)
A=(X-recoverX).^2;
fenZi=sum(A(:));
B=X.^2;
fenMu=sum(B(:));
train_error=sqrt(fenZi/fenMu);
end