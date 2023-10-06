
function dist = dist_E(x,y)
dist = [x;y];
dist = pdist(dist); % 计算各行向量之间的欧式距离
end
