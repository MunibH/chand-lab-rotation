
%%
A = [-3 3 5; 3 -4 5; 9 -7 2]./10;
[U1,S,V] = svd(A);

x = [-15:2.5:15];
y = [-15:2.5:15];

[xx,yy] = meshgrid(x,y);

z = y;

M = [];

cla
for p = 1:size(x,2)
    fprintf('.');
    for q = 1:size(y,2)
        for r = 1:size(z,2)
            U = A*[x(p) y(q) z(r)]';
            subplot(221);
            plot3(x(p), y(q), z(r),'.','color',[0.8 0.8 0.8]);
            M(p,q) = z(r);
            hold on;
            subplot(222);
            plot3(U(1), U(2), U(3),'.','color',[0.4 .4 .40]+0.4);
            hold on;
           
            
        end
    end
    drawnow;
end

OrigLineV = [];
NewLineV = [];

minV = -15;
maxV = 15;
for j=minV:1:maxV
    
    O = j*V(:,3);
    hold on;
    OrigLineV(j+abs(minV)+1,1,:) = O;
    
    O = j*V(:,1);
    OrigLineV(j+abs(minV)+1,2,:) = O; 
    
    
    D = A*j*V(:,3);
    subplot(222);
    NewLineV(j+abs(minV)+1,1,:) = D;
  
    D = A*j*V(:,1);
    NewLineV(j+abs(minV)+1,2,:) = D;
  
    hold on;
end

%%
subplot(221);
hold on;
plot3(squeeze(OrigLineV(:,1,1)),squeeze(OrigLineV(:,1,2)),squeeze(OrigLineV(:,1,3)),'r-','linewidth',2);
plot3(squeeze(OrigLineV(:,2,1)),squeeze(OrigLineV(:,2,2)),squeeze(OrigLineV(:,2,3)),'g-','linewidth',2);
axis square;

subplot(222);
hold on;
plot3(squeeze(NewLineV(:,1,1)),squeeze(NewLineV(:,1,2)),squeeze(NewLineV(:,1,3)),'r-','linewidth',2);
plot3(squeeze(NewLineV(:,2,1)),squeeze(NewLineV(:,2,2)),squeeze(NewLineV(:,2,3)),'g-','linewidth',2);