%%%%% 综合评价（10年9月18日）
clear,clc
X(:,:,1)=[0.7814    0.8900    0.9300    0.7000
    0.8047    0.6500    0.8900    1.0000
    0.7349    0.6600    0.7800    0.7000
    0.4698    0.7800    0.8200    0.5000];
X(:,:,2)=[0.8605    0.9000    0.7600    0.8000
    0.8977    0.7800    0.8800    0.9000
    0.6279    0.7600    0.8100    0.6000
    0.4651    0.8000    0.7800    0.8000];
X(:,:,3)=[0.9023    0.7800    0.6900    0.9000
    0.8558    0.7500    0.9000    0.6000
    1.0000    0.7100    0.6400    0.6000
    0.5349    0.8800    0.7200    0.9000];
X(:,:,4)=[0.8233    0.9200    0.8900    0.6000
    0.8512    0.9200    0.8900    0.7000
    0.7628    0.7000    0.6800    0.7000
       0.00001    0.5600    0.7700    0.6000];

%%%第1步，按列标准化
[m,n]=size(X(:,:,1));
t=4;
%%%%% 第2步，按列加权重

disp('w=(0.3 0.3 0.2 0.2)')

disp('属性加权矩阵为')

w=diag([0.3 0.3 0.2 0.2]);

for k=1:t
  %  X(:,:,k)=X{k};
  %  X=eval(['X' num2str(k)]);  %%%取第k个矩阵
    Y(:,:,k)=X(:,:,k)*w;
  %  temp=[reshape(Y(:,:,k)',1,[])];
  % eval(['Y',num2str(k),'=  sprintf(''%4.2f  %4.2f  %4.2f   %4.2f \n'', temp)'])
end

Y

%%% 第3步，求加权矩阵的熵

for k=1:t 
    P(:,:,k)=Y(:,:,k)./sum(sum(Y(:,:,k)));
    E(k)=-sum(sum(P(:,:,k).*log(P(:,:,k))))/log(m*n);   
end

disp('Y的熵为')
E

% Y_mean = zeros(m,n);
% for k=1:t
%     Y_mea = mean(Y(:,:,k),[],3);
% end
% disp('算术平均矩阵为')
% Y_mea


% Y_mean = zeros(m,n);
% for k=1:t
%     Y_mean = Y_mean+Y(:,:,k);
% end
% Y_mean = Y_mean/4;
% temp=[reshape(Y_mean',1,[])];
  % eval(['Y_mean''=  %%%由此算出的E_mean是1.7022
Y_mean=(Y(:,:,1)+Y(:,:,2)+Y(:,:,3)+Y(:,:,4))/4;
sprintf('%4.2f  %4.2f  %4.2f   %4.2f \n', Y_mean');
disp('算术平均矩阵为')
Y_mean  

% w=sum(sum(Y_mean))
% 
P=Y_mean./sum(sum(Y_mean));
E_mean=-sum(sum(P.*log(P)))/log(m*n) 

for k=1:t
    D(k) =abs( E(k)-E_mean);
    R(k) =E_mean./(E_mean+D(k));
end
disp('第4步，求4个偏离度')
D, R

disp('步骤8，weights of DMs 为：')

 lambda=R/sum(R) 
 
%%%% 第9步，将标准阵加决策者权
for k=1:t
    G(:,:,k)=Y(:,:,k)*lambda(k);
end
disp('第9步，加权到个人决策矩阵')
G

%%%% 转换成方案矩阵
disp('第10步，群决策矩阵')
Gi=permute(G,[3,2,1])
%%% 方案理想解

for k=1:t
    G_pos = max(Gi,[],3);
    G_neg = min(Gi,[],3);
end
disp('Positive and negative ideal decisions')
G_pos
G_neg

% for i=1:m
%     for j=1:n
%         Y_left(i,j) = min(H(i,j,:));
%         Y_right(i,j) = max(H(i,j,:));
%     end
% end
% disp('第3步，求3个理想矩阵')
% Y_left,Y_right

%  第4步，求3个偏离度
G_quare_posi=sum(sum(G_pos.^2));
G_quare_nega=sum(sum(G_neg.^2));
for k=1:t
%     D_pos(k) = norm(Hi(:,:,k)-H_pos);
%     D_neg(k) = norm(Hi(:,:,k)-H_neg);
G_G_posi(k)=sum(sum(Gi(:,:,k).*G_pos));
G_square(k)=sum(sum(Gi(:,:,k).^2));
G_G_nega(k)=sum(sum(Gi(:,:,k).*G_neg));

 NP_G_G_posi(k)=(1+min(G_square(k),G_quare_posi))./(1+max(G_square(k),G_quare_posi)+abs(G_G_posi(k)-G_quare_posi));%%G在G_{+}上标准化投影
 NP_G_G_nega(k)=(1+min(G_square(k),G_quare_nega))./(1+max(G_square(k),G_quare_nega)+abs(G_G_nega(k)-G_quare_nega));%%G在G_{-}上标准化投影
end
disp('Group utility measurements')
NP_G_G_posi, NP_G_G_nega 

% %%% 第5步，计算相对贴近度
disp('Group utility-based relative closeness')
RC_1 = NP_G_G_posi./(NP_G_G_posi+NP_G_G_nega)

%  group regret matrices
for k=1:t
    GR(:,:,k)=G_pos-Gi(:,:,k);
end
disp('Group regret matrices')
GR

for k=1:t
    GR_max = max(GR,[],3);
    GR_min = min(GR,[],3);
end

disp('Maximum and minimum group regret matrices')

GR_max
GR_min

R_max_square=sum(sum(GR_max.^2));
R_min_square=sum(sum(GR_min.^2));
for k=1:t
 R_R_max(k)=sum(sum(GR(:,:,k).*GR_max));
R_square(k)=sum(sum(GR(:,:,k).^2));
 R_R_min(k)=sum(sum(GR(:,:,k).*GR_min));
NP_R_R_max(k)=(1+min(R_square(k),R_max_square))./(1+max(R_square(k),R_max_square)+abs( R_R_max(k)-R_max_square));%% GR在GR_{max}上标准化投影 
NP_R_R_min(k)=(1+min(R_square(k),R_min_square))./(1+max(R_square(k),R_min_square)+abs( R_R_min(k)-R_min_square));%% GR在GR_{min}上标准化投影
end
disp('Group regret measurements')
NP_R_R_max
NP_R_R_min


disp('Group regret-based relative closeness')
RC_2 = NP_R_R_min./(NP_R_R_min+NP_R_R_max)

%  group  satisfaction matrices
for k=1:t
    GS(:,:,k)=Gi(:,:,k)-G_neg;
end
disp('Group satisfaction matrices')
GS

for k=1:t
    GS_max = max(GS,[],3);
    GS_min = min(GS,[],3);
end
disp('Maximum and minimum group  satisfaction matrices')
GS_max
GS_min

S_max_square=sum(sum(GS_max.^2));
S_min_square=sum(sum(GS_min.^2));
for k=1:t
 S_S_max(k)=sum(sum(GS(:,:,k).*GS_max));
 S_square(k)=sum(sum(GS(:,:,k).^2));
 S_S_min(k)=sum(sum(GS(:,:,k).*GS_min));
NP_S_S_max(k)=(1+min(S_square(k),S_max_square))./(1+max(S_square(k),S_max_square)+abs( S_S_max(k)-S_max_square));%% GR在GR_{max}上标准化投影 
NP_S_S_min(k)=(1+min(S_square(k),S_min_square))./(1+max(S_square(k),S_min_square)+abs( S_S_min(k)-S_min_square));%% GR在GR_{min}上标准化投影
end
disp('Group satisfaction measurements')
NP_S_S_max
NP_S_S_min

disp('Group satisfaction-based relative closeness')
RC_3 = NP_S_S_max./(NP_S_S_max+NP_S_S_min)

%  第7步，计算综合评价矩阵
Q_comprehensive = zeros(m,n);
for k=1:4
    Q_comprehensive = (RC_1 + RC_2+RC_3)/3;
end
disp('第7步，计算综合评价矩阵')
Q_comprehensive



    

