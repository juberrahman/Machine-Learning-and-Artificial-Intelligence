clc; 
clear variables; 
close all;

%initialize learning rate from 0.1 to 0.5 for 1st dataset

alpha = 0.1:0.1:0.5;

iteration = 100:200:500;
error1 = zeros( length(alpha), length(iteration));
data = importdata('d-10.csv');
data1 = struct2cell(data);
x1 = data1{1};

%assign perceptron weights
w0 = 1;
rng('default')
w_rest = rand(1, 10);
w = [w0 w_rest];

r = size(x1, 1);
x1 = [ones(r, 1) x1];
c = size(x1, 2) - 1;


for k = 1:length(alpha)
    for ii = 1:length(iteration)
        for i = 1:iteration(ii)
            for j = 1:r
                dot_product =  w*x1(j, 1:c)';
                if(dot_product > 0)
                    o = 1;
                else
                    o = -1;
                end
                w = w + (alpha(k)*(x1(j, c+1) - o))*x1(j, 1:c);
            end
        end
        
    output = x1( : , 1:c).*repmat(w, r, 1);
    output = sum(output, 2);
    output(find(output > 0)) = 1;
    output(find(output < 0)) = -1;

    target = x1(:, c+1);
    error1(k, ii) = sum((target - output).^2)/length(target);
    end 
  
end