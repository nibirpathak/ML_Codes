clear all
close all
addpath('Z:\Research\LCAD Sampling\NFP-Sampling\Time lapse images\03.04.19\C1\Images\results') 
load all_time.mat
load avg_control.mat
load avg_cell.mat
% F=@(x,xdata)x(1)*exp(-x(2)*xdata)+x(3);
% x0 =[1 0.04 0];
V=15;
P=25;
% F=@(x,xdata)exp(x(1)*V^(x(2))*P^(x(3))*(exp(-x(4)*V^(x(5))*P^(x(6))*xdata)-1));
% x0 =[0.9 0.136 0.087 0.0024 0.055 -0.5];

F=@(x,xdata)exp(x(1)*P^(x(2))*(exp(-x(3)*P^(x(4))*xdata)-1));
x0 =[0.3 -0.08 0.6 -0.6];
t(1,:)=t(1,:)-4.4;
T=t(1,1:45);
Intensity =avg_cell(1:45);
[x,resnorm,~,exitflag,output] = lsqcurvefit(F,x0,T',Intensity);
figure(1)
plot(T,Intensity,'o','MarkerFaceColor','b');
hold on
plot(T,F(x,T),'r','Linewidth',3)
hold off

xlabel('Time(s)');
ylabel('Normalized Mean Intensity(I)');x0=1; y0=1; width=3.5; height=3.5;
legend('Data','Fit')
legend('boxoff')
set(gcf,'units','inches','position',[x0,y0,width,height]);set(gca,'fontsize',12,'FontName','Arial')
