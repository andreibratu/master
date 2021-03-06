function ex1_2
    A = [2, 2, 0; 0, 8, 3];
    b = [5; 15];
    % A \ b is equivalent to inv(A)*b
    sprintf("Solution of Ax=b is: %s", mat2str(A \ b))

    B = A;
    % d) calculate this random c
    A(1, 2) = 4;
    c = 0;
    for i = -4:4:4
        c = c + i * A.' * b;
    end
    c
    
    A
    B
    A .* B    % Element-wise multiplication
    A.' * B   % Dumbass simple multiplication
