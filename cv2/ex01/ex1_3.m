function similar = ex1_3(x, y, eps)
    if isequal(size(x), size(y))
        eps_v = eps .* ones(size(x));
        if abs(x - y) < eps_v
            similar = 1;
        else
            similar = 0;
        end
    else
        similar = 0;
    end