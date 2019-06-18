clc
clear
mass = xlsread('masses.xlsx');
position = xlsread('positions.xlsx');
velocity = xlsread('velocities.xlsx');


m=mass;
for i=1:100
    x(i)=complex(position(i,1),position(i,2));
    v(i)=complex(velocity(i,1),velocity(i,2));
end
x=transpose(x);
v=transpose(v);

dt=0.0001;
td=0.1;
dmin=100;
G=-667000;
n=100;
one=ones(1,n);

k=0;

while dmin>td

y=x*one;
diff=transpose(y)-y;
ndiff=diff;
for i=1:n
    for j=1:n
        dist(i,j)=abs(diff(i,j));
        %dmin=min(dist(:)) 
        dmin = min(dist(dist~=0))
    end
end

for i=1:n
    for j=1:n
    if(i~=j)
    cube= abs(ndiff(i,j))^3 ; 
    ndiff(i,j)=ndiff(i,j)/cube;
    end
end
end
acc=ndiff*m*G;
x=x+v*dt+0.5*dt*dt*acc;
v=v+dt*acc;
k=k+1;
end

display(k);



















