%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Based on PCA animations shown here:
% http://stats.stackexchange.com/questions/2691
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Generate Data

clear,clc,close all
rng(42)

X = randn(100,2);
X = X*chol([1 0.6; 0.6 0.6]);
X = bsxfun(@minus, X, mean(X));

[a,b] = eig(cov(X));

%% Rotating animation

fig = figure('Position', [100 100 1000 400]);

% set(gcf,'color','w');
% axis([-3.5 3.5 -3.5 3.5])
axis square
hold on
darkBackground(fig);
for alpha = 1:1:215
    w = [cosd(alpha) sind(alpha)];
    z = X*w'*w;
    
    cla
    for i=1:100
        plot([X(i,1) z(i,1)], [X(i,2) z(i,2)], '--', 'Color',[1 1 1 0.5])
    end
    
    plot(w(1)*3.5*[-1 1], w(2)*3.5*[-1 1], 'g')
    plot(-w(2)*2*[-1 1], w(1)*2*[-1 1], 'b-')
    
    plot(z(:,1), z(:,2), 'yo','MarkerSize',5)
    plot(X(:,1), X(:,2), 'r.', 'MarkerSize',10)
    scatter(0,0,65,'k','filled', 'LineWidth', 2, 'MarkerFaceColor', [1 1 1], 'MarkerEdgeColor', [0 0 0])
    
    a1 = 3.5;
    a2 = 4.5;
    plot(a(1,2)*[-a2 -a1], a(2,2)*[-a2 -a1], 'm', 'LineWidth', 2)
    plot(a(1,2)*[ a1  a2], a(2,2)*[ a1  a2], 'm', 'LineWidth', 2)
    drawnow
    
    frame = getframe(fig);
    if alpha == 1
        [imind,map] = rgb2ind(frame.cdata,16,'nodither');
    else
        imind(:,:,1,alpha) = rgb2ind(frame.cdata,map,'nodither');
    end
end
imwrite(imind,map, 'images/animation_pca.gif', 'DelayTime', 0, 'LoopCount', inf)
















